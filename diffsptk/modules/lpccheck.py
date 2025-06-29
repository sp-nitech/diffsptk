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
from ..utils.private import check_size, filter_values
from .base import BaseFunctionalModule
from .lpc2par import LinearPredictiveCoefficientsToParcorCoefficients
from .par2lpc import ParcorCoefficientsToLinearPredictiveCoefficients


class LinearPredictiveCoefficientsStabilityCheck(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lpccheck.html>`_
    for details.

    Parameters
    ----------
    lpc_order : int >= 0
        The order of the LPC, :math:`M`.

    margin : float in (0, 1)
        The margin to guarantee the stability of LPC.

    warn_type : ['ignore', 'warn', 'exit']
        The warning type.

    """

    def __init__(
        self, lpc_order: int, margin: float = 1e-16, warn_type: str = "warn"
    ) -> None:
        super().__init__()

        self.in_dim = lpc_order + 1

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """Check the stability of the input LPC coefficients.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            The input LPC coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The modified LPC coefficients.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        tensor([-0.9966, -0.2970, -0.2173,  0.0594,  0.5831])
        >>> lpc = diffsptk.LPC(3, 5)
        >>> a = lpc(x)
        >>> a
        tensor([ 1.1528, -0.2613, -0.0274,  0.1778])
        >>> lpccheck = diffsptk.LinearPredictiveCoefficientsStabilityCheck(3)
        >>> a2 = lpccheck(a)
        tensor([ 1.1528, -0.2613, -0.0274,  0.1778])

        """
        check_size(a.size(-1), self.in_dim, "dimension of LPC")
        return self._forward(a, *self.values)

    @staticmethod
    def _func(a: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = LinearPredictiveCoefficientsStabilityCheck._precompute(
            a.size(-1) - 1, *args, **kwargs
        )
        return LinearPredictiveCoefficientsStabilityCheck._forward(a, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(lpc_order: int, margin: float) -> None:
        if lpc_order < 0:
            raise ValueError("lpc_order must be non-negative.")
        if not 0 < margin < 1:
            raise ValueError("margin must be in (0, 1).")

    @staticmethod
    def _precompute(lpc_order: int, margin: float, warn_type: str) -> Precomputed:
        LinearPredictiveCoefficientsStabilityCheck._check(lpc_order, margin)
        return (1 - margin, warn_type)

    @staticmethod
    def _forward(a: torch.Tensor, bound: float, warn_type: str) -> torch.Tensor:
        k = LinearPredictiveCoefficientsToParcorCoefficients._func(a)
        K, k1 = torch.split(k, [1, k.size(-1) - 1], dim=-1)

        if torch.any(1 <= torch.abs(k1)):
            if warn_type == "ignore":
                pass
            elif warn_type == "warn":
                warnings.warn("Detected unstable LPC coefficients.")
            elif warn_type == "exit":
                raise RuntimeError("Detected unstable LPC coefficients.")
            else:
                raise RuntimeError

        k1 = torch.clip(k1, -bound, bound)
        k2 = torch.cat((K, k1), dim=-1)
        a2 = ParcorCoefficientsToLinearPredictiveCoefficients._func(k2)
        return a2
