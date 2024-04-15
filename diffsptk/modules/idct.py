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
from .dct import DiscreteCosineTransform as DCT


class InverseDiscreteCosineTransform(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/idct.html>`_
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

    def forward(self, y):
        """Apply inverse DCT to input.

        Parameters
        ----------
        y : Tensor [shape=(..., L)]
            Input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            Inverse DCT output.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> dct = diffsptk.DCT(4)
        >>> idct = diffsptk.IDCT(4)
        >>> x2 = idct(dct(x))
        >>> x2
        tensor([-4.4703e-08,  1.0000e+00,  2.0000e+00,  3.0000e+00])

        """
        check_size(y.size(-1), self.dct_length, "dimension of input")
        return self._forward(y, self.W)

    @staticmethod
    def _forward(y, W):
        return torch.matmul(y, W)

    @staticmethod
    def _func(y):
        W = InverseDiscreteCosineTransform._precompute(
            y.size(-1), dtype=y.dtype, device=y.device
        )
        return InverseDiscreteCosineTransform._forward(y, W)

    @staticmethod
    def _precompute(dct_length, dtype=None, device=None):
        return DCT._precompute(dct_length, dtype=dtype, device=device).T
