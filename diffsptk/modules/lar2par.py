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


class LogAreaRatioToParcorCoefficients(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lar2par.html>`_
    for details.

    Parameters
    ----------
    par_order : int >= 0
        The order of the PARCOR coefficients, :math:`M`.

    """

    def __init__(self, par_order):
        super().__init__()

        self.input_dim = par_order + 1

        self.values = self._precompute(par_order)

    def forward(self, g):
        """Convert LAR to PARCOR.

        Parameters
        ----------
        g : Tensor [shape=(..., M+1)]
            The log area ratio.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The PARCOR coefficients.

        Examples
        --------
        >>> g = diffsptk.ramp(1, 4) * 0.1
        >>> lar2par = diffsptk.LogAreaRatioToParcorCoefficients(3)
        >>> k = lar2par(g)
        >>> k
        tensor([0.1000, 0.0997, 0.1489, 0.1974])

        """
        check_size(g.size(-1), self.input_dim, "dimension of parcor")
        return self._forward(g, *self.values)

    @staticmethod
    def _func(x):
        values = LogAreaRatioToParcorCoefficients._precompute(x.size(-1) - 1)
        return LogAreaRatioToParcorCoefficients._forward(x, *values)

    @staticmethod
    def _check(par_order):
        if par_order < 0:
            raise ValueError("par_order must be non-negative.")

    @staticmethod
    def _precompute(par_order):
        LogAreaRatioToParcorCoefficients._check(par_order)
        return (0.5,)

    @staticmethod
    def _forward(g, c):
        K, g = torch.split(g, [1, g.size(-1) - 1], dim=-1)
        k = torch.cat((K, torch.tanh(c * g)), dim=-1)
        return k
