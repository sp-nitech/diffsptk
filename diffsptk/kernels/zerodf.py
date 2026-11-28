# ------------------------------------------------------------------------ #
# Copyright 2022 SPTK Working Group                                        #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ------------------------------------------------------------------------ #

from typing import cast

import torch

from .common import launch, prepare

_SOURCE = "zerodf.cu"
_KERNEL_NAMES = ("zerodf_forward", "zerodf_backward_x", "zerodf_backward_b")

# Number of consecutive samples processed by a thread.
_BLOCKING = 8
# Number of threads in a block and number of samples staged at once in the
# backward kernel with respect to x.
_BWD_X_THREADS = 32
_BWD_X_CHUNK = 256
_DEFINES = (
    ("BLOCKING", _BLOCKING),
    ("BWD_X_THREADS", _BWD_X_THREADS),
    ("BWD_X_CHUNK", _BWD_X_CHUNK),
)


def _n_groups(n: int) -> int:
    return (n + _BLOCKING - 1) // _BLOCKING


def _n_tiles(n: int) -> int:
    span = _BWD_X_THREADS * _BLOCKING
    return (n + span - 1) // span


# The output y = F(x, b) is bilinear in x and b. Let T(g, x, b) = <g, F(x, b)>.
# Then F, the gradient with respect to x (Gx), and the gradient with respect to b
# (Gb) are the partial derivatives of the trilinear form T:
#
#   F(x, b) = dT/dg,   Gx(g, b) = dT/dx,   Gb(g, x) = dT/db.
#
# Hence, the derivatives of each of the three operations are given by the other
# two, which makes the operations differentiable to any order.


def _forward(x, b, ramp, zeroth_index):
    B, T = x.shape
    N, L = b.shape[-2:]
    P = ramp.numel()
    y = torch.empty_like(x)
    launch(
        _SOURCE,
        "zerodf_forward",
        B * N * _n_groups(P),
        [x, b, ramp, y, B, T, N, L, P, zeroth_index],
        dtype=x.dtype,
        device=x.device,
        defines=_DEFINES,
    )
    return y


def _backward_x(g, b, ramp, zeroth_index):
    B, T = g.shape
    N, L = b.shape[-2:]
    P = ramp.numel()
    grad_x = torch.empty_like(g)
    launch(
        _SOURCE,
        "zerodf_backward_x",
        B * _n_tiles(T) * _BWD_X_THREADS,
        [g, b, ramp, grad_x, B, T, N, L, P, zeroth_index],
        dtype=g.dtype,
        device=g.device,
        defines=_DEFINES,
        block_size=_BWD_X_THREADS,
    )
    return grad_x


def _backward_b(g, x, ramp, zeroth_index, L):
    B, T = x.shape
    P = ramp.numel()
    N = T // P
    grad_b = torch.empty(B, N, L, dtype=x.dtype, device=x.device)
    launch(
        _SOURCE,
        "zerodf_backward_b",
        B * N * _n_groups(L),
        [g, x, ramp, grad_b, B, T, N, L, P, zeroth_index],
        dtype=x.dtype,
        device=x.device,
        defines=_DEFINES,
    )
    return grad_b


class _Forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b, ramp, zeroth_index):
        ctx.save_for_backward(x, b, ramp)
        ctx.zeroth_index = zeroth_index
        return _forward(x, b, ramp, zeroth_index)

    @staticmethod
    def backward(ctx, *grad):
        x, b, ramp = ctx.saved_tensors
        z = ctx.zeroth_index
        g = grad[0].contiguous()
        grad_x = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_x = _BackwardX.apply(g, b, ramp, z)
        if ctx.needs_input_grad[1]:
            grad_b = _BackwardB.apply(g, x, ramp, z, b.size(-1))
        return grad_x, grad_b, None, None


class _BackwardX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, b, ramp, zeroth_index):
        ctx.save_for_backward(g, b, ramp)
        ctx.zeroth_index = zeroth_index
        return _backward_x(g, b, ramp, zeroth_index)

    @staticmethod
    def backward(ctx, *grad):
        g, b, ramp = ctx.saved_tensors
        z = ctx.zeroth_index
        h = grad[0].contiguous()
        grad_g = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_g = _Forward.apply(h, b, ramp, z)
        if ctx.needs_input_grad[1]:
            grad_b = _BackwardB.apply(g, h, ramp, z, b.size(-1))
        return grad_g, grad_b, None, None


class _BackwardB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, x, ramp, zeroth_index, L):
        ctx.save_for_backward(g, x, ramp)
        ctx.zeroth_index = zeroth_index
        return _backward_b(g, x, ramp, zeroth_index, L)

    @staticmethod
    def backward(ctx, *grad):
        g, x, ramp = ctx.saved_tensors
        z = ctx.zeroth_index
        h = grad[0].contiguous()
        grad_g = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_g = _Forward.apply(x, h, ramp, z)
        if ctx.needs_input_grad[1]:
            grad_x = _BackwardX.apply(g, h, ramp, z)
        return grad_g, grad_x, None, None, None


def zerodf(
    x: torch.Tensor, b: torch.Tensor, ramp: torch.Tensor, zeroth_index: int
) -> torch.Tensor:
    """Apply a time-varying all-zero filter using the CUDA kernel.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        The input signal.

    b : Tensor [shape=(..., T/P, L)]
        The filter coefficients.

    ramp : Tensor [shape=(P,)]
        The interpolation weights within a frame.

    zeroth_index : int
        The index of the zeroth coefficient.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        The output signal.

    Raises
    ------
    KernelUnavailableError
        If the kernels cannot be compiled.

    """
    if x.shape[:-1] != b.shape[:-2]:
        raise ValueError("The batch shapes of x and b must be the same.")

    # Compile all kernels in advance so that a compilation failure is detected
    # before the autograd graph is built.
    prepare(_SOURCE, _KERNEL_NAMES, dtype=x.dtype, device=x.device, defines=_DEFINES)

    y = _Forward.apply(
        x.reshape(-1, x.size(-1)).contiguous(),
        b.reshape(-1, b.size(-2), b.size(-1)).contiguous(),
        ramp.to(dtype=x.dtype).contiguous().view(-1),
        zeroth_index,
    )
    return cast(torch.Tensor, y).view(x.shape)
