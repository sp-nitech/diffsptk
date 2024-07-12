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
from .dst import DiscreteSineTransform as DST


class InverseDiscreteSineTransform(nn.Module):
    """This is the opposite module to :func:`~diffsptk.DiscreteSineTransform`.

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

    def forward(self, y):
        """Apply inverse DST to input.

        Parameters
        ----------
        y : Tensor [shape=(..., L)]
            Input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            Inverse DST output.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> dst = diffsptk.DST(4)
        >>> idst = diffsptk.IDST(4)
        >>> x2 = idst(dst(x))
        >>> x2
        tensor([1.1921e-07, 1.0000e+00, 2.0000e+00, 3.0000e+00])

        """
        check_size(y.size(-1), self.dst_length, "dimension of input")
        return self._forward(y, self.W)

    @staticmethod
    def _forward(y, W):
        return torch.matmul(y, W)

    @staticmethod
    def _func(y):
        W = InverseDiscreteSineTransform._precompute(
            y.size(-1), dtype=y.dtype, device=y.device
        )
        return InverseDiscreteSineTransform._forward(y, W)

    @staticmethod
    def _precompute(dst_length, dtype=None, device=None):
        return DST._precompute(dst_length, dtype=dtype, device=device).T
