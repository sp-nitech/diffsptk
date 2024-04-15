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
from .delta import Delta


class MaximumLikelihoodParameterGeneration(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mlpg.html>`_
    for details. Currently, only global unit variance is supported.

    Parameters
    ----------
    size : int >= 1
        Length of input, :math:`T`.

    seed : list[list[float]] or list[int]
        Delta coefficients or width(s) of 1st (and 2nd) regression coefficients.

    """

    def __init__(self, size, seed=[[-0.5, 0, 0.5], [1, -2, 1]]):
        super().__init__()

        assert 1 <= size

        self.size = size

        self.register_buffer("M", self._precompute(size, seed))

    def forward(self, u):
        """Perform MLPG to obtain smoothed static sequence.

        Parameters
        ----------
        u : Tensor [shape=(..., T, DxH)]
            Time-variant mean vectors with delta components.

        Returns
        -------
        out : Tensor [shape=(..., T, D)]
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
        check_size(u.size(-2), self.size, "length of input")
        return self._forward(u, self.M)

    @staticmethod
    def _forward(mean, M):
        T = mean.size(-2)
        H = M.size(-1) // T
        u = mean.reshape(*mean.shape[:-2], T * H, -1)
        c = torch.einsum("...Td,tT->...td", u, M)
        return c

    @staticmethod
    def _func(u, seed):
        M = MaximumLikelihoodParameterGeneration._precompute(
            u.size(-2), seed, dtype=u.dtype, device=u.device
        )
        return MaximumLikelihoodParameterGeneration._forward(u, M)

    @staticmethod
    def _precompute(size, seed, dtype=None, device=None):
        # Make window.
        window = Delta._precompute(seed, True, dtype=torch.double, device=device)

        # Compute threshold.
        if isinstance(seed[0], (tuple, list)):
            th = [0] + [len(coefficients) // 2 for coefficients in seed]
        else:
            th = [0] + list(seed)
        th = torch.tensor(th, dtype=torch.double, device=device).unsqueeze(1)

        H, L = window.shape
        N = (L - 1) // 2
        T = size
        W = torch.zeros((T * H, T), dtype=torch.double, device=device)

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
        WSW = torch.matmul(WS, W)
        WSW = torch.linalg.inv(WSW)
        M = torch.matmul(WSW, WS)  # (T, TxH)
        return to(M, dtype=dtype)
