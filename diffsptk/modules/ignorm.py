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
from .gnorm import GeneralizedCepstrumGainNormalization


class GeneralizedCepstrumInverseGainNormalization(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ignorm.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    gamma : float in [-1, 1]
        The gamma parameter, :math:`\\gamma`.

    c : int >= 1 or None
        The number of filter stages.

    References
    ----------
    .. [1] T. Kobayashi et al., "Spectral analysis using generalized cepstrum," *IEEE
           Transactions on Acoustics, Speech, and Signal Processing*, vol. 32, no. 5,
           pp. 1087-1089, 1984.

    """

    def __init__(self, cep_order, gamma=0, c=None):
        super().__init__()

        self.in_dim = cep_order + 1

        self.values = self._precompute(cep_order, gamma, c)

    def forward(self, y):
        """Perform cepstrum inverse gain normalization.

        Parameters
        ----------
        y : Tensor [shape=(..., M+1)]
            The normalized generalized cepstrum.

        Returns
        -------
        x : Tensor [shape=(..., M+1)]
            The generalized cepstrum.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 4)
        >>> gnorm = diffsptk.GeneralizedCepstrumGainNormalization(3, c=2)
        >>> ignorm = diffsptk.GeneralizedCepstrumInverseGainNormalization(3, c=2)
        >>> x2 = ignorm(gnorm(x))
        >>> x2
        tensor([1., 2., 3., 4.])

        """
        check_size(y.size(-1), self.in_dim, "dimension of cepstrum")
        return self._forward(y, *self.values)

    @staticmethod
    def _func(y, *args, **kwargs):
        values = GeneralizedCepstrumInverseGainNormalization._precompute(
            y.size(-1) - 1, *args, **kwargs
        )
        return GeneralizedCepstrumInverseGainNormalization._forward(y, *values)

    @staticmethod
    def _check(*args, **kwargs):
        return GeneralizedCepstrumGainNormalization._check(*args, **kwargs)

    @staticmethod
    def _precompute(*args, **kwargs):
        return GeneralizedCepstrumGainNormalization._precompute(*args, **kwargs)

    @staticmethod
    def _forward(y, gamma):
        K, y = torch.split(y, [1, y.size(-1) - 1], dim=-1)
        if gamma == 0:
            x0 = torch.log(K)
            x1 = y
        else:
            z = torch.pow(K, gamma)
            x0 = (z - 1) / gamma
            x1 = y * z
        x = torch.cat((x0, x1), dim=-1)
        return x
