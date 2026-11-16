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

import importlib

import pytest
import torch

import diffsptk
from diffsptk.kernels import common as kernel_common
from diffsptk.kernels.zerodf import zerodf as cuda_zerodf

zerodf_module = importlib.import_module("diffsptk.modules.zerodf")


requires_cuda_kernel = pytest.mark.skipif(
    not kernel_common.is_available(),
    reason="CUDA kernel compilation is not available.",
)


def tolerance(dtype):
    if dtype == torch.double:
        return {"rtol": 1e-10, "atol": 1e-10}
    return {"rtol": 1e-4, "atol": 1e-4}


CUDA_CASES = [
    # (M, P, zeroth_index)
    (0, 1, 0),
    (5, 1, 0),
    (3, 10, 0),
    (3, 10, 3),
    (3, 10, 1),
    (20, 5, 0),
    (20, 5, 20),
    (20, 5, 7),
    (199, 80, 0),
    (1999, 80, 0),
    (1999, 80, 1999),
    (20, 300, 7),
    (600, 300, 0),
    (600, 300, 600),
]


def make_cuda_inputs(M, P, dtype, batch_shape=(2,), N=7, requires_grad=False):
    params = {"device": "cuda", "dtype": dtype, "requires_grad": requires_grad}
    x = torch.randn(*batch_shape, N * P, **params)
    b = torch.randn(*batch_shape, N, M + 1, **params)
    return x, b


def make_zerodf_pair(M, P, zeroth_index, dtype, ignore_gain=False):
    params = {
        "filter_order": M,
        "frame_period": P,
        "zeroth_index": zeroth_index,
        "ignore_gain": ignore_gain,
        "device": "cuda",
        "dtype": dtype,
    }
    return (
        diffsptk.AllZeroDigitalFilter(**params, mode="efficient"),
        diffsptk.AllZeroDigitalFilter(**params, mode="cuda"),
    )


@requires_cuda_kernel
@pytest.mark.parametrize("M, P, zeroth_index", CUDA_CASES)
def test_cuda_forward(dtype, M, P, zeroth_index):
    ref, cuda = make_zerodf_pair(M, P, zeroth_index, dtype)
    x, b = make_cuda_inputs(M, P, dtype)
    assert kernel_common.is_supported(x, b)
    torch.testing.assert_close(cuda(x, b), ref(x, b), **tolerance(dtype))


@requires_cuda_kernel
@pytest.mark.parametrize("M, P, zeroth_index", [(3, 10, 0), (3, 10, 3)])
def test_cuda_forward_ignore_gain(dtype, M, P, zeroth_index):
    ref, cuda = make_zerodf_pair(M, P, zeroth_index, dtype, ignore_gain=True)
    x, b = make_cuda_inputs(M, P, dtype)
    b = b + 3  # Keep the gain away from zero.
    torch.testing.assert_close(cuda(x, b), ref(x, b), **tolerance(dtype))


@requires_cuda_kernel
@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
def test_cuda_forward_shape(dtype, batch_shape, M=4, P=5):
    ref, cuda = make_zerodf_pair(M, P, 0, dtype)
    x, b = make_cuda_inputs(M, P, dtype, batch_shape=batch_shape)
    y = cuda(x, b)
    assert y.shape == x.shape
    torch.testing.assert_close(y, ref(x, b), **tolerance(dtype))


@requires_cuda_kernel
@pytest.mark.parametrize(
    "M, P, zeroth_index", [(0, 1, 0), (3, 2, 0), (3, 2, 3), (3, 2, 1)]
)
def test_cuda_gradcheck(M, P, zeroth_index):
    _, cuda = make_zerodf_pair(M, P, zeroth_index, torch.double)
    x, b = make_cuda_inputs(M, P, torch.double, N=4, requires_grad=True)
    assert torch.autograd.gradcheck(cuda, (x, b))


@requires_cuda_kernel
@pytest.mark.parametrize(
    "M, P, zeroth_index", [(0, 1, 0), (3, 2, 0), (3, 2, 3), (3, 2, 1)]
)
def test_cuda_gradgradcheck(M, P, zeroth_index):
    _, cuda = make_zerodf_pair(M, P, zeroth_index, torch.double)
    x, b = make_cuda_inputs(M, P, torch.double, N=4, requires_grad=True)
    assert torch.autograd.gradgradcheck(cuda, (x, b))


@requires_cuda_kernel
@pytest.mark.parametrize("M, P, zeroth_index", CUDA_CASES)
@pytest.mark.parametrize("wrt", ["both", "x", "b"])
def test_cuda_backward(dtype, M, P, zeroth_index, wrt):
    ref, cuda = make_zerodf_pair(M, P, zeroth_index, dtype)
    x, b = make_cuda_inputs(M, P, dtype)
    x.requires_grad_(wrt in ("both", "x"))
    b.requires_grad_(wrt in ("both", "b"))
    inputs = [t for t in (x, b) if t.requires_grad]
    dy = torch.randn_like(x)

    grads_ref = torch.autograd.grad(ref(x, b), inputs, dy)
    grads = torch.autograd.grad(cuda(x, b), inputs, dy)
    for g, g_ref in zip(grads, grads_ref, strict=True):
        torch.testing.assert_close(g, g_ref, **tolerance(dtype))


@pytest.mark.parametrize("ignore_gain", [False, True])
def test_cuda_fallback_on_cpu(monkeypatch, ignore_gain, M=3, P=10, N=7):
    monkeypatch.setattr(zerodf_module, "cuda_zerodf", _raise_if_called)
    x = torch.randn(2, N * P)
    b = torch.randn(2, N, M + 1) + 3
    params = {"filter_order": M, "frame_period": P, "ignore_gain": ignore_gain}
    ref = diffsptk.AllZeroDigitalFilter(**params, mode="efficient")
    cuda = diffsptk.AllZeroDigitalFilter(**params, mode="cuda")
    torch.testing.assert_close(cuda(x, b), ref(x, b), rtol=0, atol=0)


def test_cuda_fallback_without_compiler(monkeypatch, device, M=3, P=10, N=7):
    monkeypatch.delattr(torch.cuda, "_compile_kernel", raising=False)
    monkeypatch.setattr(zerodf_module, "cuda_zerodf", _raise_if_called)
    assert not kernel_common.is_available()
    x = torch.randn(2, N * P, device=device)
    b = torch.randn(2, N, M + 1, device=device)
    ref = diffsptk.AllZeroDigitalFilter(M, P, mode="efficient", device=device)
    cuda = diffsptk.AllZeroDigitalFilter(M, P, mode="cuda", device=device)
    torch.testing.assert_close(cuda(x, b), ref(x, b), rtol=0, atol=0)


@requires_cuda_kernel
@pytest.mark.parametrize("x_shape, b_shape", [((2, 70), (7, 4)), ((70,), (2, 7, 4))])
def test_cuda_batch_mismatch(monkeypatch, x_shape, b_shape):
    monkeypatch.setattr(zerodf_module, "cuda_zerodf", _raise_if_called)
    x = torch.randn(x_shape, device="cuda")
    b = torch.randn(b_shape, device="cuda")
    cuda = diffsptk.AllZeroDigitalFilter(3, 10, mode="cuda", device="cuda")
    with pytest.raises(RuntimeError):
        cuda(x, b)


@requires_cuda_kernel
def test_cuda_kernel_rejects_batch_mismatch():
    x = torch.randn(70, device="cuda")
    b = torch.randn(2, 7, 4, device="cuda")
    ramp = torch.arange(10, device="cuda") / 10
    with pytest.raises(ValueError):
        cuda_zerodf(x, b, ramp, 0)


@requires_cuda_kernel
@pytest.mark.parametrize(
    "failing_kernel", ["zerodf_forward", "zerodf_backward_x", "zerodf_backward_b"]
)
def test_cuda_fallback_on_compile_error(monkeypatch, failing_kernel, M=3, P=10):
    original = torch.cuda._compile_kernel

    def compile_kernel(source, name, *args, **kwargs):
        if name == failing_kernel:
            raise RuntimeError("Compilation failed.")
        return original(source, name, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "_compile_kernel", compile_kernel)
    monkeypatch.setattr(kernel_common, "_KERNELS", {})
    monkeypatch.setattr(kernel_common, "_disabled", False, raising=False)

    ref, cuda = make_zerodf_pair(M, P, 0, torch.double)
    x, b = make_cuda_inputs(M, P, torch.double, requires_grad=True)
    with pytest.warns(UserWarning, match="CUDA kernel"):
        y = cuda(x, b)
    y_ref = ref(x, b)
    torch.testing.assert_close(y, y_ref, rtol=0, atol=0)
    grads = torch.autograd.grad(y.sum(), (x, b))
    grads_ref = torch.autograd.grad(y_ref.sum(), (x, b))
    for g, g_ref in zip(grads, grads_ref, strict=True):
        torch.testing.assert_close(g, g_ref, rtol=0, atol=0)
    assert not kernel_common.is_available()


def _raise_if_called(*args, **kwargs):
    raise AssertionError("The CUDA kernel must not be launched.")
