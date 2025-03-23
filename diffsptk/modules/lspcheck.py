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

import warnings

import torch

from ..typing import Precomputed
from ..utils.private import check_size, get_values
from .base import BaseFunctionalModule


class LineSpectralPairsStabilityCheck(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lspcheck.html>`_
    for details.

    Parameters
    ----------
    lsp_order : int >= 0
        The order of the LSP, :math:`M`.

    rate : float in [0, 1]
        The rate of distance between two adjacent LSPs.

    n_iter : int >= 0
        The number of iterations for the modification.

    warn_type : ['ignore', 'warn', 'exit']
        The warning type.

    """

    def __init__(
        self, lsp_order: int, rate: float = 0, n_iter: int = 1, warn_type: str = "warn"
    ) -> None:
        super().__init__()

        self.in_dim = lsp_order + 1

        self.values = self._precompute(*get_values(locals()))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Check the stability of the input LSP coefficients.

        Parameters
        ----------
        w : Tensor [shape=(..., M+1)]
            The input LSP coefficients in radians.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The modified LSP coefficients in radians.

        Examples
        --------
        >>> w1 = torch.tensor([0, 0, 1]) * torch.pi
        >>> lspcheck = diffsptk.LineSpectralPairsStabilityCheck(2, rate=0.01)
        >>> w2 = lspcheck(w1)
        >>> w2
        tensor([0.0000, 0.0105, 3.1311])

        """
        check_size(w.size(-1), self.in_dim, "dimension of LSP")
        return self._forward(w, *self.values)

    @staticmethod
    def _func(w: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = LineSpectralPairsStabilityCheck._precompute(
            w.size(-1) - 1, *args, **kwargs
        )
        return LineSpectralPairsStabilityCheck._forward(w, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(lsp_order: int, rate: float, n_iter: int) -> None:
        if lsp_order < 0:
            raise ValueError("lsp_order must be non-negative.")
        if not 0 <= rate <= 1:
            raise ValueError("rate must be in [0, 1].")
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative.")

    @staticmethod
    def _precompute(
        lsp_order: int, rate: float, n_iter: int, warn_type: str
    ) -> Precomputed:
        LineSpectralPairsStabilityCheck._check(lsp_order, rate, n_iter)
        return (
            rate * torch.pi / (lsp_order + 1),
            n_iter,
            warn_type,
        )

    @staticmethod
    def _forward(
        w: torch.Tensor, min_distance: float, n_iter: int, warn_type: str
    ) -> torch.Tensor:
        K, w1 = torch.split(w, [1, w.size(-1) - 1], dim=-1)

        distance = torch.diff(w1, dim=-1)
        if torch.any(distance <= 0) or torch.any(w <= 0) or torch.any(torch.pi <= w):
            if warn_type == "ignore":
                pass
            elif warn_type == "warn":
                warnings.warn("Detected unstable LSP coefficients.")
            elif warn_type == "exit":
                raise RuntimeError("Detected unstable LSP coefficients.")
            else:
                raise RuntimeError

        w1 = w1.clone()
        for _ in range(n_iter):
            for m in range(w1.size(-1) - 1):
                n = m + 1
                distance = w1[..., n] - w1[..., m]
                step_size = 0.5 * torch.clip(min_distance - distance, min=0)
                w1[..., m] -= step_size
                w1[..., n] += step_size
            w1 = torch.clip(w1, min=min_distance, max=torch.pi - min_distance)
            distance = torch.diff(w1, dim=-1)
            if torch.all(min_distance - 1e-16 <= distance):
                break

        w2 = torch.cat((K, w1), dim=-1)
        return w2
