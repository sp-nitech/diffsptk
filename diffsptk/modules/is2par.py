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

from ..misc.utils import check_size
from .base import BaseFunctionalModule


class InverseSineToParcorCoefficients(BaseFunctionalModule):
    """This is a similar module to :func:`~diffsptk.LogAreaRatioToParcorCoefficients`.

    Parameters
    ----------
    par_order : int >= 0
        The order of the PARCOR coefficients, :math:`M`.

    """

    def __init__(self, par_order):
        super().__init__()

        self.in_dim = par_order + 1

        self.values = self._precompute(par_order)

    def forward(self, s):
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
        >>> s = diffsptk.ramp(1, 4) * 0.1
        >>> is2par = diffsptk.InverseSineToParcorCoefficients(3)
        >>> k = is2par(s)
        >>> k
        tensor([0.1000, 0.3090, 0.4540, 0.5878])

        """
        check_size(s.size(-1), self.in_dim, "dimension of parcor")
        return self._forward(s, *self.values)

    @staticmethod
    def _func(x):
        values = InverseSineToParcorCoefficients._precompute(x.size(-1) - 1)
        return InverseSineToParcorCoefficients._forward(x, *values)

    @staticmethod
    def _check(par_order):
        if par_order < 0:
            raise ValueError("par_order must be non-negative.")

    @staticmethod
    def _precompute(par_order):
        InverseSineToParcorCoefficients._check(par_order)
        return (torch.pi / 2,)

    @staticmethod
    def _forward(s, c):
        K, s = torch.split(s, [1, s.size(-1) - 1], dim=-1)
        k = torch.cat((K, torch.sin(c * s)), dim=-1)
        return k
