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

from ..utils.private import check_size
from ..utils.private import get_values
from .base import BaseFunctionalModule


class MelCepstrumInversePowerNormalization(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ipnorm.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    """

    def __init__(self, cep_order):
        super().__init__()

        self.in_dim = cep_order + 2

        self.values = self._precompute(*get_values(locals()))

    def forward(self, y):
        """Perform mel-cepstrum inverse power normalization.

        Parameters
        ----------
        y : Tensor [shape=(..., M+2)]
            The log power and power-normalized cepstrum.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The output cepstrum.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 4)
        >>> pnorm = diffsptk.MelCepstrumPowerNormalization(3, alpha=0.1)
        >>> ipnorm = diffsptk.MelCepstrumInversePowerNormalization(3)
        >>> y = ipnorm(pnorm(x))
        >>> y
        tensor([1., 2., 3., 4.])

        """
        check_size(y.size(-1), self.in_dim, "dimension of cepstrum")
        return self._forward(y)

    @staticmethod
    def _func(y):
        MelCepstrumInversePowerNormalization._precompute(y.size(-1) - 1)
        return MelCepstrumInversePowerNormalization._forward(y)

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(cep_order):
        if cep_order < 0:
            raise ValueError("cep_order must be non-negative.")

    @staticmethod
    def _precompute(cep_order):
        MelCepstrumInversePowerNormalization._check(cep_order)

    @staticmethod
    def _forward(y):
        P, y1, y2 = torch.split(y, [1, 1, y.size(-1) - 2], dim=-1)
        x = torch.cat((0.5 * P + y1, y2), dim=-1)
        return x
