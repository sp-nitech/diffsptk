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

from .c2acr import CepstrumToAutocorrelation
from .freqt import FrequencyTransform


class MelCepstrumPowerNormalization(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/pnorm.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        Order of cepstrum, :math:`M`.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    ir_length : int >= 1
        Length of impulse response.

    """

    def __init__(self, cep_order, alpha=0, ir_length=128):
        super().__init__()

        self.mc2pow = nn.Sequential(
            FrequencyTransform(cep_order, ir_length - 1, -alpha),
            CepstrumToAutocorrelation(ir_length - 1, 0, ir_length),
        )

    def forward(self, x):
        """Perform cepstrum power normalization.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            Input cepstrum.

        Returns
        -------
        out : Tensor [shape=(..., M+2)]
            Power-normalized cepstrum.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 4)
        >>> pnorm = diffsptk.MelCepstrumPowerNormalization(3, alpha=0.1)
        >>> y = pnorm(x)
        >>> y
        tensor([ 8.2942, -7.2942,  2.0000,  3.0000,  4.0000])

        """
        return self._forward(x, self.mc2pow)

    @staticmethod
    def _forward(x, mc2pow):
        x0, x1 = torch.split(x, [1, x.size(-1) - 1], dim=-1)
        P = torch.log(mc2pow(x))
        y = torch.cat((P, x0 - 0.5 * P, x1), dim=-1)
        return y

    @staticmethod
    def _func(x, alpha, ir_length):
        def mc2pow(mc):
            c = FrequencyTransform._func(mc, ir_length - 1, -alpha)
            r = CepstrumToAutocorrelation._func(c, 0, ir_length)
            return r

        return MelCepstrumPowerNormalization._forward(x, mc2pow)
