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
from torch import nn

from ..misc.utils import check_size
from ..misc.utils import get_gamma


class GeneralizedCepstrumGainNormalization(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/gnorm.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        Order of cepstrum, :math:`M`.

    gamma : float in [-1, 1]
        Gamma, :math:`\\gamma`.

    c : int >= 1 or None
        Number of stages.

    """

    def __init__(self, cep_order, gamma=0, c=None):
        super().__init__()

        assert 0 <= cep_order
        assert abs(gamma) <= 1

        self.cep_order = cep_order
        self.gamma = self._precompute(gamma, c)

    def forward(self, x):
        """Perform cepstrum gain normalization.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            Generalized cepstrum.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            Normalized generalized cepstrum.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 4)
        >>> gnorm = diffsptk.GeneralizedCepstrumGainNormalization(3, c=2)
        >>> y = gnorm(x)
        >>> y
        tensor([2.2500, 1.3333, 2.0000, 2.6667])

        """
        check_size(x.size(-1), self.cep_order + 1, "dimension of cepstrum")
        return self._forward(x, self.gamma)

    @staticmethod
    def _forward(x, gamma):
        x0, x1 = torch.split(x, [1, x.size(-1) - 1], dim=-1)
        if gamma == 0:
            K = torch.exp(x0)
            y = x1
        else:
            z = 1 + gamma * x0
            K = torch.pow(z, 1 / gamma)
            y = x1 / z
        y = torch.cat((K, y), dim=-1)
        return y

    @staticmethod
    def _func(x, gamma, c=None):
        gamma = GeneralizedCepstrumGainNormalization._precompute(gamma, c)
        return GeneralizedCepstrumGainNormalization._forward(x, gamma)

    @staticmethod
    def _precompute(gamma, c):
        return get_gamma(gamma, c)
