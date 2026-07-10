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

from ..utils.private import check_size, filter_values
from .base import BaseFunctionalModule, Precomputed


class ParcorCoefficientsToLogAreaRatio(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/par2lar.html>`_
    for details.

    Parameters
    ----------
    par_order : int >= 0
        The order of the PARCOR coefficients, :math:`M`.

    """

    _takes_input_size = True

    def __init__(self, par_order: int) -> None:
        super().__init__()

        self.in_dim = par_order + 1

        self._register_precomputed(self._precompute(**filter_values(locals())))

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Convert PARCOR to LAR.

        Parameters
        ----------
        k : Tensor [shape=(..., M+1)]
            The PARCOR coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The log area ratio.

        Examples
        --------
        >>> import diffsptk
        >>> par2lar = diffsptk.ParcorCoefficientsToLogAreaRatio(3)
        >>> lar2par = diffsptk.LogAreaRatioToParcorCoefficients(3)
        >>> k = diffsptk.ramp(1, 4) * 0.1
        >>> k2 = lar2par(par2lar(k))
        >>> k2
        tensor([0.1000, 0.2000, 0.3000, 0.4000])

        """
        check_size(k.size(-1), self.in_dim, "dimension of parcor")
        return self._call_forward(k)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _p = ParcorCoefficientsToLogAreaRatio._precompute(
            x.size(-1) - 1, *args, **kwargs
        )
        return ParcorCoefficientsToLogAreaRatio._apply_precomputed(_p, k=x)

    @staticmethod
    def _check(par_order: int) -> None:
        if par_order < 0:
            raise ValueError("par_order must be non-negative.")

    @staticmethod
    def _precompute(par_order: int) -> Precomputed:
        ParcorCoefficientsToLogAreaRatio._check(par_order)
        return Precomputed(values={"c": 2})

    @staticmethod
    def _forward(k: torch.Tensor, *, c: float) -> torch.Tensor:
        K, k = torch.split(k, [1, k.size(-1) - 1], dim=-1)
        g = torch.cat((K, c * torch.atanh(k)), dim=-1)
        return g
