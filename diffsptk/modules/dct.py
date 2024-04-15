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


class DiscreteCosineTransform(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/dct.html>`_
    for details.

    Parameters
    ----------
    dct_length : int >= 1
        DCT length, :math:`L`.

    """

    def __init__(self, dct_length):
        super().__init__()

        assert 1 <= dct_length

        self.dct_length = dct_length
        self.register_buffer("W", self._precompute(self.dct_length))

    def forward(self, x):
        """Apply DCT to input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            DCT output.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> dct = diffsptk.DCT(4)
        >>> y = dct(x)
        >>> y
        tensor([ 3.0000, -2.2304,  0.0000, -0.1585])

        """
        check_size(x.size(-1), self.dct_length, "dimension of input")
        return self._forward(x, self.W)

    @staticmethod
    def _forward(x, W):
        return torch.matmul(x, W)

    @staticmethod
    def _func(x):
        W = DiscreteCosineTransform._precompute(
            x.size(-1), dtype=x.dtype, device=x.device
        )
        return DiscreteCosineTransform._forward(x, W)

    @staticmethod
    def _precompute(length, dtype=None, device=None):
        L = length
        k = torch.arange(L, dtype=torch.double, device=device)
        n = (k + 0.5) * (torch.pi / L)
        z = torch.sqrt(torch.clip(1 + k, 1, 2) / L)
        W = z.unsqueeze(0) * torch.cos(k.unsqueeze(0) * n.unsqueeze(1))
        return to(W, dtype=dtype)
