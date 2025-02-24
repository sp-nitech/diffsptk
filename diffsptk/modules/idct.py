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

from ..misc.utils import check_size
from .base import BaseFunctionalModule
from .dct import DiscreteCosineTransform as DCT


class InverseDiscreteCosineTransform(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/idct.html>`_
    for details.

    Parameters
    ----------
    dct_length : int >= 1
        The DCT length, :math:`L`.

    dct_type : int in [1, 4]
        The DCT type.

    """

    def __init__(self, dct_length, dct_type=2):
        super().__init__()

        self.in_dim = dct_length

        _, tensors = self._precompute(dct_length, dct_type)
        self.register_buffer("W", tensors[0])

    def forward(self, y):
        """Apply inverse DCT to the input.

        Parameters
        ----------
        y : Tensor [shape=(..., L)]
            The input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            The inverse DCT output.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> dct = diffsptk.DCT(4)
        >>> idct = diffsptk.IDCT(4)
        >>> x2 = idct(dct(x))
        >>> x2
        tensor([-4.4703e-08,  1.0000e+00,  2.0000e+00,  3.0000e+00])

        """
        check_size(y.size(-1), self.in_dim, "dimension of input")
        return self._forward(y, **self._buffers)

    @staticmethod
    def _func(y, *args, **kwargs):
        _, tensors = InverseDiscreteCosineTransform._precompute(
            y.size(-1), *args, **kwargs, dtype=y.dtype, device=y.device
        )
        return InverseDiscreteCosineTransform._forward(y, *tensors)

    @staticmethod
    def _check(*args, **kwargs):
        DCT._check(*args, **kwargs)

    @staticmethod
    def _precompute(dct_length, dct_type, dtype=None, device=None):
        type2type = {1: 1, 2: 3, 3: 2, 4: 4}
        return DCT._precompute(
            dct_length, type2type[dct_type], dtype=dtype, device=device
        )

    @staticmethod
    def _forward(y, W):
        return torch.matmul(y, W)
