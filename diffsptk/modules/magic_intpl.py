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

import torch

from ..typing import Precomputed
from ..utils.private import UNVOICED_SYMBOL, filter_values
from .base import BaseFunctionalModule


class MagicNumberInterpolation(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/magic_intpl.html>`_
    for details.

    Parameters
    ----------
    magic_number : float
        The magic number to be interpolated.

    """

    def __init__(self, magic_number: float = UNVOICED_SYMBOL) -> None:
        super().__init__()

        _, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("magic_number", tensors[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate magic number.

        Parameters
        ----------
        x : Tensor [shape=(B, N, D) or (N, D) or (N,)]
            The data containing magic number.

        Returns
        -------
        out : Tensor [shape=(B, N, D) or (N, D) or (N,)]
            The data after interpolation.

        Examples
        --------
        >>> import diffsptk
        >>> magic_intpl = diffsptk.MagicNumberInterpolation(0)
        >>> x = torch.tensor([0, 1, 2, 0, 4, 0]).float()
        >>> x
        tensor([0., 1., 2., 0., 4., 0.])
        >>> y = magic_intpl(x)
        >>> y
        tensor([1., 1., 2., 3., 4., 4.])

        """
        return self._forward(x, **self._buffers)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _, tensors = MagicNumberInterpolation._precompute(
            *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return MagicNumberInterpolation._forward(x, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check() -> None:
        pass

    @staticmethod
    def _precompute(
        magic_number: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        MagicNumberInterpolation._check()
        magic_number = torch.tensor(magic_number, device=device, dtype=dtype)
        return None, None, (magic_number,)

    @staticmethod
    def _forward(x: torch.Tensor, magic_number: torch.Tensor) -> torch.Tensor:
        return MagicNumberInterpolationImpl.apply(x, magic_number)


class MagicNumberInterpolationImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, magic_number):
        ctx.save_for_backward(x, magic_number)

        # Pass through if magic number is not found.
        if torch.all(x != magic_number):
            return x

        d = x.dim()
        if d == 1:
            x = x.view(1, -1, 1)
        elif d == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("Input must be 1D, 2D, or 3D tensor.")
        B, T, D = x.shape

        def compute_lerp_inputs(x, magic_number):
            is_magic_number = x == magic_number

            starts = []
            ends = []
            weights = []
            for i in range(x.size(0)):
                uniques, counts = torch.unique_consecutive(
                    is_magic_number[i],
                    return_inverse=False,
                    return_counts=True,
                    dim=-1,
                )
                w = torch.repeat_interleave(
                    uniques.to(x.dtype) / (counts + 1), counts, dim=-1
                )
                if uniques[0]:
                    w[..., : counts[0]] = 0
                w = torch.cumsum(w, dim=-1)
                w = w - torch.cummax(w * ~is_magic_number[i], dim=-1)[0]
                if uniques[0]:
                    w[..., : counts[0]] = 1
                if uniques[-1]:
                    w[..., -counts[-1] :] = 0

                uniques, indices = torch.unique_consecutive(
                    x[i],
                    return_inverse=True,
                    return_counts=False,
                    dim=-1,
                )
                pos = uniques == magic_number
                uniques[pos] = torch.roll(uniques, 1, dims=-1)[pos]
                s = uniques[indices]
                uniques[pos] = torch.roll(uniques, -1, dims=-1)[pos]
                e = uniques[indices]

                starts.append(s)
                ends.append(e)
                weights.append(w)

            starts = torch.stack(starts)
            ends = torch.stack(ends)
            weights = torch.stack(weights)
            return starts, ends, weights

        x = x.transpose(-2, -1).reshape(B * D, T)
        starts, ends, weights = compute_lerp_inputs(x, magic_number)
        y = torch.lerp(starts, ends, weights)
        y = y.reshape(B, D, T).transpose(-2, -1)

        if d == 1:
            y = y.view(-1)
        elif d == 2:
            y = y.squeeze(0)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, magic_number = ctx.saved_tensors
        return grad_output * (x != magic_number), None
