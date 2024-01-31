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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..misc.utils import cexp
from ..misc.utils import check_size
from ..misc.utils import clog
from ..misc.utils import numpy_to_torch
from .freqt import FrequencyTransform
from .gnorm import GeneralizedCepstrumGainNormalization as GainNormalization
from .ignorm import (
    GeneralizedCepstrumInverseGainNormalization as InverseGainNormalization,
)


class GeneralizedCepstrumToGeneralizedCepstrum(nn.Module):
    """Generalized cepstral transformation module.

    Parameters
    ----------
    in_order : int >= 0 [scalar]
        Order of input cepstrum, :math:`M_1`.

    out_order : int >= 0 [scalar]
        Order of output cepstrum, :math:`M_2`.

    in_gamma : float [-1 <= in_gamma <= 1]
        Input gamma, :math:`\\gamma_1`.

    out_gamma : float [-1 <= out_gamma <= 1]
        Output gamma, :math:`\\gamma_2`.

    n_fft : int >> :math:`M_1, M_2` [scalar]
        Number of FFT bins. Accurate conversion requires the large value.

    """

    def __init__(self, in_order, out_order, in_gamma, out_gamma, n_fft=512):
        super(GeneralizedCepstrumToGeneralizedCepstrum, self).__init__()

        self.in_order = in_order
        self.out_order = out_order
        self.in_gamma = in_gamma
        self.out_gamma = out_gamma
        self.n_fft = n_fft

        assert 0 <= self.in_order
        assert 0 <= self.out_order
        assert abs(self.in_gamma) <= 1
        assert abs(self.out_gamma) <= 1
        assert max(self.in_order, self.out_order) + 1 < self.n_fft

    def forward(self, c1):
        """Perform generalized cepstral transformation.

        Parameters
        ----------
        c1 : Tensor [shape=(..., M1+1)]
            Input cepstrum.

        Returns
        -------
        c2 : Tensor [shape=(..., M2+1)]
            Output cepstrum.

        """
        check_size(c1.size(-1), self.in_order + 1, "dimension of cepstrum")

        c01 = F.pad(c1[..., 1:], (1, 0))
        C1 = torch.fft.fft(c01, n=self.n_fft)

        if self.in_gamma == 0:
            sC1 = cexp(C1)
        else:
            C1 *= self.in_gamma
            C1.real += 1
            r = C1.abs() ** (1 / self.in_gamma)
            theta = C1.angle() / self.in_gamma
            sC1 = torch.polar(r, theta)

        if self.out_gamma == 0:
            C2 = clog(sC1)
        else:
            r = sC1.abs() ** self.out_gamma
            theta = sC1.angle() * self.out_gamma
            C2 = (r * torch.cos(theta) - 1) / self.out_gamma

        c02 = torch.fft.ifft(C2)[..., : self.out_order + 1].real
        c2 = torch.cat((c1[..., :1], 2 * c02[..., 1:]), dim=-1)
        return c2


class GammaDivision(nn.Module):
    def __init__(self, cep_order, gamma):
        super(GammaDivision, self).__init__()
        g = np.full(cep_order + 1, 1 / gamma)
        g[0] = 1
        self.register_buffer("g", numpy_to_torch(g))

    def forward(self, c):
        return c * self.g


class GammaMultiplication(nn.Module):
    def __init__(self, cep_order, gamma):
        super(GammaMultiplication, self).__init__()
        g = np.full(cep_order + 1, gamma)
        g[0] = 1
        self.register_buffer("g", numpy_to_torch(g))

    def forward(self, c):
        return c * self.g


class ZerothGammaDivision(nn.Module):
    def __init__(self, cep_order, gamma):
        super(ZerothGammaDivision, self).__init__()
        self.cep_order = cep_order
        self.g = 1 / gamma

    def forward(self, c):
        c0, c1 = torch.split(c, [1, self.cep_order], dim=-1)
        c0 = (c0 - 1) * self.g
        return torch.cat((c0, c1), dim=-1)


class ZerothGammaMultiplication(nn.Module):
    def __init__(self, cep_order, gamma):
        super(ZerothGammaMultiplication, self).__init__()
        self.cep_order = cep_order
        self.g = gamma

    def forward(self, c):
        c0, c1 = torch.split(c, [1, self.cep_order], dim=-1)
        c0 = c0 * self.g + 1
        return torch.cat((c0, c1), dim=-1)


class MelGeneralizedCepstrumToMelGeneralizedCepstrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mgc2mgc.html>`_
    for details. The conversion uses FFT instead of recursive formula.

    Parameters
    ----------
    in_order : int >= 0 [scalar]
        Order of input cepstrum, :math:`M_1`.

    out_order : int >= 0 [scalar]
        Order of output cepstrum, :math:`M_2`.

    in_alpha : float [-1 < in_alpha < 1]
        Input alpha, :math:`\\alpha_1`.

    out_alpha : float [-1 < out_alpha < 1]
        Output alpha, :math:`\\alpha_2`.

    in_gamma : float [-1 <= in_gamma <= 1]
        Input gamma, :math:`\\gamma_1`.

    out_gamma : float [-1 <= out_gamma <= 1]
        Output gamma, :math:`\\gamma_2`.

    in_norm : bool [scalar]
        If True, assume normalized input.

    out_norm : bool [scalar]
        If True, assume normalized output.

    in_mul : bool [scalar]
        If True, assume gamma-multiplied input.

    out_mul : bool [scalar]
        If True, assume gamma-multiplied output.

    n_fft : int >> :math:`M_1, M_2` [scalar]
        Number of FFT bins. Accurate conversion requires the large value.

    """

    def __init__(
        self,
        in_order,
        out_order,
        in_alpha=0,
        out_alpha=0,
        in_gamma=0,
        out_gamma=0,
        in_norm=False,
        out_norm=False,
        in_mul=False,
        out_mul=False,
        n_fft=512,
    ):
        super(MelGeneralizedCepstrumToMelGeneralizedCepstrum, self).__init__()

        assert not (0 == in_gamma and in_mul)

        modules = []
        if not in_norm and in_mul:
            modules.append(ZerothGammaDivision(in_order, in_gamma))

        alpha = (out_alpha - in_alpha) / (1 - in_alpha * out_alpha)
        if 0 == alpha:
            if in_order == out_order and in_gamma == out_gamma:
                if not in_mul and out_mul:
                    modules.append(GammaMultiplication(in_order, in_gamma))
                if not in_norm and out_norm:
                    modules.append(GainNormalization(in_order, in_gamma))
                if in_norm and not out_norm:
                    modules.append(InverseGainNormalization(out_order, out_gamma))
                if in_mul and not out_mul:
                    modules.append(GammaDivision(out_order, out_gamma))
            else:
                if in_mul:
                    modules.append(GammaDivision(in_order, in_gamma))
                if not in_norm:
                    modules.append(GainNormalization(in_order, in_gamma))
                if True:
                    modules.append(
                        GeneralizedCepstrumToGeneralizedCepstrum(
                            in_order, out_order, in_gamma, out_gamma, n_fft
                        )
                    )
                if not out_norm:
                    modules.append(InverseGainNormalization(out_order, out_gamma))
                if out_mul:
                    modules.append(GammaMultiplication(out_order, out_gamma))
        else:
            if in_mul:
                modules.append(GammaDivision(in_order, in_gamma))
            if in_norm:
                modules.append(InverseGainNormalization(in_order, in_gamma))
            if True:
                modules.append(FrequencyTransform(in_order, out_order, alpha))
            if out_norm or in_gamma != out_gamma:
                modules.append(GainNormalization(out_order, in_gamma))
            if in_gamma != out_gamma:
                modules.append(
                    GeneralizedCepstrumToGeneralizedCepstrum(
                        out_order, out_order, in_gamma, out_gamma, n_fft
                    )
                )
            if not out_norm and in_gamma != out_gamma:
                modules.append(InverseGainNormalization(out_order, out_gamma))
            if out_mul:
                modules.append(GammaMultiplication(out_order, out_gamma))

        if not out_norm and out_mul:
            modules.append(ZerothGammaMultiplication(out_order, out_gamma))

        self.seq = nn.Sequential(*modules)

    def forward(self, c1):
        """Convert mel-generalized cepstrum to mel-generalized cepstrum.

        Parameters
        ----------
        c1 : Tensor [shape=(..., M1+1)]
            Input mel-cepstrum.

        Returns
        -------
        c2 : Tensor [shape=(..., M2+1)]
            Converted mel-cepstrum.

        Examples
        --------
        >>> c1 = diffsptk.ramp(3)
        >>> mgc2mgc = diffsptk.MelGeneralizedCepstrumToMelGeneralizedCepstrum(3, 4, 0.1)
        >>> c2 = mgc2mgc(c1)
        >>> c2
        tensor([-0.0830,  0.6831,  1.1464,  3.1334,  0.9063])

        """
        c2 = self.seq(c1)
        return c2
