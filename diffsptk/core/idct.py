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
import torch.nn as nn

from ..misc.utils import numpy_to_torch
from .dct import make_dct_matrix


class InverseDiscreteCosineTransform(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/idct.html>`_
    for details.

    Parameters
    ----------
    dct_length : int >= 1 [scalar]
        DCT length, :math:`L`.

    """

    def __init__(self, dct_length):
        super(InverseDiscreteCosineTransform, self).__init__()

        assert 1 <= dct_length

        W = make_dct_matrix(dct_length)
        self.register_buffer("W", numpy_to_torch(W.T))

    def forward(self, y):
        """Apply inverse DCT to input.

        Parameters
        ----------
        y : Tensor [shape=(..., L)]
            Input.

        Returns
        -------
        x : Tensor [shape=(..., L)]
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
        x = torch.matmul(y, self.W)
        return x
