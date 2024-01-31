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

import numpy as np
import torch
import torch.nn as nn

from ..misc.utils import numpy_to_torch


def make_dct_matrix(L):
    """Make DCT matrix.

    Parameters
    ----------
    L : int >= 1 [scalar]
        DCT length, :math:`L`.

    Returns
    -------
    W : ndarray [shape=(L, L)]
        DCT matrix.

    """
    W = np.empty((L, L))
    n = (np.arange(L) + 0.5) * (np.pi / L)
    c = np.sqrt(2 / L)
    for k in range(L):
        z = np.sqrt(1 / L) if k == 0 else c
        W[:, k] = z * np.cos(k * n)
    return W


class DiscreteCosineTransform(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/dct.html>`_
    for details.

    Parameters
    ----------
    dct_length : int >= 1 [scalar]
        DCT length, :math:`L`.

    """

    def __init__(self, dct_length):
        super(DiscreteCosineTransform, self).__init__()

        assert 1 <= dct_length

        W = make_dct_matrix(dct_length)
        self.register_buffer("W", numpy_to_torch(W))

    def forward(self, x):
        """Apply DCT to input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Input.

        Returns
        -------
        y : Tensor [shape=(..., L)]
            DCT output.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> dct = diffsptk.DCT(4)
        >>> y = dct(x)
        >>> y
        tensor([ 3.0000, -2.2304,  0.0000, -0.1585])

        """
        y = torch.matmul(x, self.W)
        return y
