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


class InverseSineToParcorCoefficients(BaseFunctionalModule):
    """This is a similar module to :func:`~diffsptk.LogAreaRatioToParcorCoefficients`.

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

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Convert IS to PARCOR.

        Parameters
        ----------
        s : Tensor [shape=(..., M+1)]
            The inverse sine coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The PARCOR coefficients.

        Examples
        --------
        >>> import diffsptk
        >>> is2par = diffsptk.InverseSineToParcorCoefficients(3)
        >>> s = diffsptk.ramp(1, 4) * 0.1
        >>> k = is2par(s)
        >>> k
        tensor([0.1000, 0.3090, 0.4540, 0.5878])

        """
        check_size(s.size(-1), self.in_dim, "dimension of parcor")
        return self._call_forward(s)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _p = InverseSineToParcorCoefficients._precompute(
            x.size(-1) - 1, *args, **kwargs
        )
        return InverseSineToParcorCoefficients._apply_precomputed(_p, s=x)

    @staticmethod
    def _check(par_order: int) -> None:
        if par_order < 0:
            raise ValueError("par_order must be non-negative.")

    @staticmethod
    def _precompute(par_order: int) -> Precomputed:
        InverseSineToParcorCoefficients._check(par_order)
        return Precomputed(values={"c": torch.pi / 2})

    @staticmethod
    def _forward(s: torch.Tensor, *, c: float) -> torch.Tensor:
        K, s = torch.split(s, [1, s.size(-1) - 1], dim=-1)
        k = torch.cat((K, torch.sin(c * s)), dim=-1)
        return k
