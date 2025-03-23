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

from ..typing import Precomputed
from ..utils.private import check_size, get_values
from .base import BaseFunctionalModule
from .lpc2par import LinearPredictiveCoefficientsToParcorCoefficients


class ParcorCoefficientsToLinearPredictiveCoefficients(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/par2lpc.html>`_
    for details.

    Parameters
    ----------
    lpc_order : int >= 0
        The order of the LPC, :math:`M`.

    gamma : float in [-1, 1]
        The gamma parameter, :math:`\\gamma`.

    c : int >= 1 or None
        The number of filter stages.

    """

    def __init__(self, lpc_order: int, gamma: float = 1, c: int | None = None) -> None:
        super().__init__()

        self.in_dim = lpc_order + 1

        self.values = ParcorCoefficientsToLinearPredictiveCoefficients._precompute(
            *get_values(locals())
        )

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Convert PARCOR to LPC.

        Parameters
        ----------
        k : Tensor [shape=(..., M+1)]
            The PARCOR coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The LPC coefficients.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([ 0.7829, -0.2028,  1.6912,  0.1454,  0.4861])
        >>> lpc = diffsptk.LPC(3, 5)
        >>> a = lpc(x)
        >>> a
        tensor([ 1.6036,  0.0573, -0.5615, -0.0638])
        >>> lpc2par = diffsptk.LinearPredictiveCoefficientsToParcorCoefficients(3)
        >>> par2lpc = diffsptk.ParcorCoefficientsToLinearPredictiveCoefficients(3)
        >>> a2 = par2lpc(lpc2par(a))
        >>> a2
        tensor([ 1.6036,  0.0573, -0.5615, -0.0638])

        """
        check_size(k.size(-1), self.in_dim, "dimension of PARCOR")
        return self._forward(k, *self.values)

    @staticmethod
    def _func(k: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = ParcorCoefficientsToLinearPredictiveCoefficients._precompute(
            k.size(-1) - 1, *args, **kwargs
        )
        return ParcorCoefficientsToLinearPredictiveCoefficients._forward(k, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(*args, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def _precompute(*args, **kwargs) -> Precomputed:
        return LinearPredictiveCoefficientsToParcorCoefficients._precompute(
            *args, **kwargs
        )

    @staticmethod
    def _forward(k: torch.Tensor, gamma: float) -> torch.Tensor:
        a = k / gamma
        for m in range(2, k.size(-1)):
            km = k[..., m : m + 1]
            am = a[..., 1:m]
            a[..., 1:m] = am + km * am.flip(-1)
        return a
