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
from torch import nn

from ..misc.utils import UNVOICED_SYMBOL


class MagicNumberInterpolation(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/magic_intpl.html>`_
    for details.

    Parameters
    ----------
    magic_number : float
        Magic number.

    """

    def __init__(self, magic_number=UNVOICED_SYMBOL):
        super().__init__()

        self.register_buffer("magic_number", self._precompute(magic_number))

    def forward(self, x):
        """Interpolate magic number.

        Parameters
        ----------
        x : Tensor [shape=(B, N, D) or (N, D) or (N,)]
            Data containing magic number.

        Returns
        -------
        out : Tensor [shape=(B, N, D) or (N, D) or (N,)]
            Data after interpolation.

        Examples
        --------
        >>> x = torch.tensor([0, 1, 2, 0, 4, 0]).float()
        >>> x
        tensor([0., 1., 2., 0., 4., 0.])
        >>> magic_intpl = diffsptk.MagicNumberInterpolation(0)
        >>> y = magic_intpl(x)
        >>> y
        tensor([1., 1., 2., 3., 4., 4.])

        """
        return self._forward(x, self.magic_number)

    @staticmethod
    def _forward(x, magic_number):
        return MagicNumberInterpolationImpl.apply(x, magic_number)

    @staticmethod
    def _func(x, magic_number):
        magic_number = MagicNumberInterpolation._precompute(
            magic_number, dtype=x.dtype, device=x.device
        )
        return MagicNumberInterpolation._forward(x, magic_number)

    @staticmethod
    def _precompute(magic_number, dtype=None, device=None):
        if not torch.is_tensor(magic_number):
            magic_number = torch.tensor(magic_number, dtype=dtype, device=device)
        return magic_number


class MagicNumberInterpolationImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, magic_number):
        ctx.save_for_backward(x, magic_number)

        # Pass through if magic number is not found
        if torch.all(x != magic_number):
            return x

        d = x.dim()
        if d == 1:
            x = x.view(1, -1, 1)
        elif d == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, "Input must be 3D tensor"
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
                w = torch.repeat_interleave(uniques / (counts + 1), counts, dim=-1)
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
