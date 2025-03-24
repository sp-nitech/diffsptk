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


class ParcorCoefficientsToInverseSine(BaseFunctionalModule):
    """This is a similar module to :func:`~diffsptk.ParcorCoefficientsToLogAreaRatio`.

    Parameters
    ----------
    par_order : int >= 0
        The order of the PARCOR coefficients, :math:`M`.

    """

    def __init__(self, par_order):
        super().__init__()

        self.in_dim = par_order + 1

        self.values = self._precompute(*get_values(locals()))

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Convert PARCOR to IS.

        Parameters
        ----------
        k : Tensor [shape=(..., M+1)]
            The PARCOR coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The inverse sine coefficients.

        Examples
        --------
        >>> k = diffsptk.ramp(1, 4) * 0.1
        >>> par2is = diffsptk.ParcorCoefficientsToInverseSine(3)
        >>> is2par = diffsptk.InverseSineToParcorCoefficients(3)
        >>> k2 = is2par(par2is(k))
        >>> k2
        tensor([0.1000, 0.2000, 0.3000, 0.4000])

        """
        check_size(k.size(-1), self.in_dim, "dimension of parcor")
        return self._forward(k, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = ParcorCoefficientsToInverseSine._precompute(
            x.size(-1) - 1, *args, **kwargs
        )
        return ParcorCoefficientsToInverseSine._forward(x, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(par_order: int) -> None:
        if par_order < 0:
            raise ValueError("par_order must be non-negative.")

    @staticmethod
    def _precompute(par_order: int) -> Precomputed:
        ParcorCoefficientsToInverseSine._check(par_order)
        return (2 / torch.pi,)

    @staticmethod
    def _forward(k: torch.Tensor, c: float) -> torch.Tensor:
        K, k = torch.split(k, [1, k.size(-1) - 1], dim=-1)
        eps = 1e-6
        k = torch.clip(k, min=-1 + eps, max=1 - eps)
        s = torch.cat((K, c * torch.asin(k)), dim=-1)
        return s
