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

from ..misc.utils import to
from .b2mc import MLSADigitalFilterCoefficientsToMelCepstrum
from .c2acr import CepstrumToAutocorrelation
from .freqt import FrequencyTransform
from .mc2b import MelCepstrumToMLSADigitalFilterCoefficients


class MelCepstrumPostfiltering(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mcpf.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        Order of mel-cepstrum, :math:`M`.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    beta : float
        Intensity parameter, :math:`\\beta`.

    onset : int >= 0
        Onset index.

    ir_length : int >= 1
        Length of impulse response.

    """

    def __init__(self, cep_order, alpha=0, beta=0, onset=2, ir_length=128):
        super().__init__()

        assert 0 <= onset

        self.mc2en = nn.Sequential(
            FrequencyTransform(cep_order, ir_length - 1, -alpha),
            CepstrumToAutocorrelation(ir_length - 1, 0, ir_length),
        )
        self.mc2b = MelCepstrumToMLSADigitalFilterCoefficients(cep_order, alpha)
        self.b2mc = MLSADigitalFilterCoefficientsToMelCepstrum(cep_order, alpha)

        weight = torch.full((cep_order + 1,), 1 + beta)
        weight[:onset] = 1
        self.register_buffer("weight", to(weight))

    def forward(self, mc):
        """Perform mel-cesptrum postfiltering.

        Parameters
        ----------
        mc : Tensor [shape=(..., M+1)]
            Mel-cepstral coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            Postfiltered mel-cepstral coefficients.

        Examples
        --------
        >>> X = diffsptk.nrand(4).square()
        >>> X
        tensor([0.2725, 2.5650, 0.3552, 0.3757, 0.1904])
        >>> mcep = diffsptk.MelCepstralAnalysis(3, 8, 0.1)
        >>> mcpf = diffsptk.MelCepstrumPostfiltering(3, 0.1, 0.2)
        >>> mc1 = mcep(X)
        >>> mc1
        tensor([-0.2819,  0.3486, -0.2487, -0.3600])
        >>> mc2 = mcpf(mc1)
        >>> mc2
        tensor([-0.3256,  0.3486, -0.2984, -0.4320])

        """
        mc1 = mc
        e1 = self.mc2en(mc1)

        mc2 = mc * self.weight
        e2 = self.mc2en(mc2)

        b2 = self.mc2b(mc2)
        b2[..., :1] += 0.5 * torch.log(e1 / e2)
        mc2 = self.b2mc(b2)
        return mc2
