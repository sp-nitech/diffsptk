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
from .delta import make_window


class MaximumLikelihoodParameterGeneration(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mlpg.html>`_
    for details. Currently, only global unit variance is supported.

    Parameters
    ----------
    size : int >= 1 [scalar]
        Length of input, :math:`T`.

    seed : list[list[float]] or list[int]
        Delta coefficients or width(s) of 1st (and 2nd) regression coefficients.

    """

    def __init__(self, size, seed=[[-0.5, 0, 0.5], [1, -2, 1]]):
        super(MaximumLikelihoodParameterGeneration, self).__init__()

        assert 1 <= size

        # Make window.
        window = make_window(seed)

        # Compute threshold.
        if isinstance(seed[0], (tuple, list)):
            th = [0] + [len(coefficients) // 2 for coefficients in seed]
        else:
            th = [0] + list(seed)
        th = np.expand_dims(np.asarray(th), 1)

        H, L = window.shape
        N = (L - 1) // 2
        T = size
        W = np.zeros((T * H, T))

        # Make window matrix
        for t in range(T):
            hs = H * t
            he = hs + H
            ts = t - N
            te = ts + L
            if ts < 0:
                W[hs:he, :te] = window[:, -ts:] * (th <= t)
            elif T < te:
                W[hs:he, ts:] = window[:, : T - ts] * (th < T - t)
            else:
                W[hs:he, ts:te] = window

        WS = W.T  # Assume unit variance.
        WSW = np.matmul(WS, W)
        WSW = np.linalg.inv(WSW)
        M = np.matmul(WSW, WS)  # (T, TxH)
        self.register_buffer("M", numpy_to_torch(M))

        # Save number of windows.
        self.H = H

    def forward(self, mean):
        """Perform MLPG to obtain static sequence.

        Parameters
        ----------
        mean : Tensor [shape=(..., T, DxH)]
            Time-variant mean vectors with delta components.

        Returns
        -------
        c : Tensor [shape=(..., T, D)]
            Static components.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 8).view(1, -1, 2)
        >>> x
        tensor([[[1., 2.],
                 [3., 4.],
                 [5., 6.],
                 [7., 8.]]])
        >>> delta = diffsptk.Delta([[-0.5, 0], [0, 0, 0.5]])
        >>> y = delta(x)
        >>> y
        tensor([[[ 1.0000,  2.0000, -0.5000, -1.0000,  1.5000,  2.0000],
                 [ 3.0000,  4.0000, -0.5000, -1.0000,  2.5000,  3.0000],
                 [ 5.0000,  6.0000, -1.5000, -2.0000,  3.5000,  4.0000],
                 [ 7.0000,  8.0000, -2.5000, -3.0000,  3.5000,  4.0000]]])
        >>> mlpg = diffsptk.MLPG(y.size(1), [[-0.5, 0], [0, 0, 0.5]])
        >>> c = mlpg(y)
        >>> c
        tensor([[[1., 2.],
                 [3., 4.],
                 [5., 6.],
                 [7., 8.]]])

        """
        T = mean.size(-2)
        u = mean.reshape(*mean.shape[:-2], T * self.H, -1)
        c = torch.einsum("...Td,tT->...td", u, self.M)
        return c
