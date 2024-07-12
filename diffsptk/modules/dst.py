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

from ..misc.utils import check_size
from ..misc.utils import to


class DiscreteSineTransform(nn.Module):
    """Discrete sine transform module.

    Parameters
    ----------
    dst_length : int >= 1
        DST length, :math:`L`.

    """

    def __init__(self, dst_length):
        super().__init__()

        assert 1 <= dst_length

        self.dst_length = dst_length
        self.register_buffer("W", self._precompute(self.dst_length))

    def forward(self, x):
        """Apply DST to input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            DST output.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> dst = diffsptk.DST(4)
        >>> y = dst(x)
        >>> y
        tensor([ 2.7716, -2.0000,  1.1481, -1.0000])

        """
        check_size(x.size(-1), self.dst_length, "dimension of input")
        return self._forward(x, self.W)

    @staticmethod
    def _forward(x, W):
        return torch.matmul(x, W)

    @staticmethod
    def _func(x):
        W = DiscreteSineTransform._precompute(
            x.size(-1), dtype=x.dtype, device=x.device
        )
        return DiscreteSineTransform._forward(x, W)

    @staticmethod
    def _precompute(length, dtype=None, device=None):
        L = length
        k = torch.arange(L, dtype=torch.double, device=device) + 1
        n = (torch.pi / L) * (k - 0.5)
        z = torch.sqrt(torch.clip(k, min=1, max=2) / L).flip(0)
        W = z.unsqueeze(0) * torch.sin(k.unsqueeze(0) * n.unsqueeze(1))
        return to(W, dtype=dtype)
