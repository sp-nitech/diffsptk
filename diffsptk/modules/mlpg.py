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

from ..typing import ArrayLike, Precomputed
from ..utils.private import check_size, get_values, to
from .base import BaseFunctionalModule
from .delta import Delta


class MaximumLikelihoodParameterGeneration(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mlpg.html>`_
    for details. Currently, only global unit variance is supported.

    Parameters
    ----------
    size : int >= 1
        The length of the input sequence, :math:`T`.

    seed : list[list[float]] or list[int]
        The delta coefficients or the width(s) of 1st (and 2nd) regression coefficients.

    """

    def __init__(
        self,
        size: int,
        seed: ArrayLike[ArrayLike[float]] | ArrayLike[int] = [
            [-0.5, 0, 0.5],
            [1, -2, 1],
        ],
    ) -> None:
        super().__init__()

        self.in_length = size

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("M", tensors[0])

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Perform MLPG given the mean vectors with delta components.

        Parameters
        ----------
        u : Tensor [shape=(..., T, DxH)]
            The time-variant mean vectors with delta components.

        Returns
        -------
        out : Tensor [shape=(..., T, D)]
            The smoothed static components.

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
        check_size(u.size(-2), self.in_length, "length of input")
        return self._forward(u, **self._buffers)

    @staticmethod
    def _func(u: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _, tensors = MaximumLikelihoodParameterGeneration._precompute(
            u.size(-2), *args, **kwargs, device=u.device, dtype=u.dtype
        )
        return MaximumLikelihoodParameterGeneration._forward(u, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check() -> None:
        pass

    @staticmethod
    def _precompute(
        size: int,
        seed: ArrayLike[ArrayLike[float]] | ArrayLike[int],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        MaximumLikelihoodParameterGeneration._check()

        # Make window.
        window = Delta._precompute(seed, True, device=device, dtype=torch.double)[-1][0]

        # Compute threshold.
        if isinstance(seed[0], (tuple, list)):
            th = [0] + [len(coefficients) // 2 for coefficients in seed]
        else:
            th = [0] + list(seed)
        th = torch.tensor(th, device=device, dtype=torch.double).unsqueeze(1)

        H, L = window.shape
        N = (L - 1) // 2
        T = size
        W = torch.zeros((T * H, T), device=device, dtype=torch.double)

        # Make window matrix.
        # codespell:ignore-begin
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
        # codespell:ignore-end

        WS = W.T  # Assume unit variance.
        WSW = torch.matmul(WS, W)
        WSW = torch.linalg.inv(WSW)
        M = torch.matmul(WSW, WS)  # (T, TxH)
        return None, None, (to(M, dtype=dtype),)

    @staticmethod
    def _forward(mean: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        T = mean.size(-2)
        H = M.size(-1) // T
        u = mean.reshape(*mean.shape[:-2], T * H, -1)
        c = torch.einsum("...Td,tT->...td", u, M)
        return c
