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

import math

import torch
import torch.nn as nn

from .mgc2mgc import MelGeneralizedCepstrumToMelGeneralizedCepstrum


class MelGeneralizedCepstrumToSpectrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mgc2sp.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0 [scalar]
        Order of mel-cepstrum, :math:`M`.

    fft_length : int >= 2 [scalar]
        Number of FFT bins, :math:`L`.

    alpha : float [-1 < alpha < 1]
        Warping factor, :math:`\\alpha`.

    gamma : float [-1 <= gamma <= 1]
        Gamma, :math:`\\gamma`.

    norm : bool [scalar]
        If True, assume normalized cepstrum.

    mul : bool [scalar]
        If True, assume gamma-multiplied cepstrum.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power', \
                  'cycle', 'radian', 'degree', 'complex']
        Output format.

    n_fft : int >> :math:`L` [scalar]
        Number of FFT bins. Accurate conversion requires the large value.

    """

    def __init__(
        self,
        cep_order,
        fft_length,
        alpha=0,
        gamma=0,
        norm=False,
        mul=False,
        out_format="power",
        n_fft=512,
    ):
        super(MelGeneralizedCepstrumToSpectrum, self).__init__()

        self.fft_length = fft_length

        assert 2 <= self.fft_length

        if out_format == 0 or out_format == "db":
            c = 20 / math.log(10)
            self.convert = lambda x: x.real * c
        elif out_format == 1 or out_format == "log-magnitude":
            self.convert = lambda x: x.real
        elif out_format == 2 or out_format == "magnitude":
            self.convert = lambda x: torch.exp(x.real)
        elif out_format == 3 or out_format == "power":
            self.convert = lambda x: torch.exp(2 * x.real)
        elif out_format == 4 or out_format == "cycle":
            self.convert = lambda x: x.imag / math.pi
        elif out_format == 5 or out_format == "radian":
            self.convert = lambda x: x.imag
        elif out_format == 6 or out_format == "degree":
            c = 180 / math.pi
            self.convert = lambda x: x.imag * c
        elif out_format == "complex":
            self.convert = lambda x: torch.polar(torch.exp(x.real), x.imag)
        else:
            raise ValueError(f"out_format {out_format} is not supported")

        self.mgc2c = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
            cep_order,
            fft_length // 2,
            in_alpha=alpha,
            in_gamma=gamma,
            in_norm=norm,
            in_mul=mul,
            n_fft=n_fft,
        )

    def forward(self, mc):
        """Convert mel-cepstrum to spectrum.

        Parameters
        ----------
        mc : Tensor [shape=(..., M+1)]
            Mel-cepstrum.

        Returns
        -------
        sp : Tensor [shape=(..., L/2+1)]
            Amplitude spectrum or phase spectrum.

        Examples
        --------
        >>> x = diffsptk.ramp(19)
        >>> stft = diffsptk.STFT(frame_length=10, frame_period=10, fft_length=16)
        >>> mcep = diffsptk.MelCepstralAnalysis(3, 16, 0.1, n_iter=1)
        >>> mc = mcep(stft(x))
        >>> mc
        tensor([[-0.8851,  0.7917, -0.1737,  0.0175],
                [-0.3522,  4.4222, -1.0882, -0.0511]])
        >>> mc2sp = diffsptk.MelGeneralizedCepstrumToSpectrum(3, 8, 0.1)
        >>> sp = mc2sp(mc)
        >>> sp
        tensor([[6.0634e-01, 4.6702e-01, 1.7489e-01, 4.4821e-02, 2.3869e-02],
                [3.5677e+02, 1.9435e+02, 6.0078e-01, 2.4278e-04, 8.8537e-06]])

        """
        c = self.mgc2c(mc)
        sp = torch.fft.rfft(c, n=self.fft_length)
        sp = self.convert(sp)
        return sp
