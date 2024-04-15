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
from torch import nn

from ..misc.utils import check_size
from .mgc2mgc import MelGeneralizedCepstrumToMelGeneralizedCepstrum


class MelGeneralizedCepstrumToSpectrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mgc2sp.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        Order of mel-cepstrum, :math:`M`.

    fft_length : int >= 2
        Number of FFT bins, :math:`L`.

    alpha : float in (-1, 1)
        Warping factor, :math:`\\alpha`.

    gamma : float in [-1, 1]
        Gamma, :math:`\\gamma`.

    norm : bool
        If True, assume normalized cepstrum.

    mul : bool
        If True, assume gamma-multiplied cepstrum.

    n_fft : int >> L
        Number of FFT bins. Accurate conversion requires the large value.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power', \
                  'cycle', 'radian', 'degree', 'complex']
        Output format.

    """

    def __init__(
        self,
        cep_order,
        fft_length,
        *,
        alpha=0,
        gamma=0,
        norm=False,
        mul=False,
        n_fft=512,
        out_format="power",
    ):
        super().__init__()

        self.cep_order = cep_order

        self.mgc2c = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
            cep_order,
            fft_length // 2,
            in_alpha=alpha,
            in_gamma=gamma,
            in_norm=norm,
            in_mul=mul,
            n_fft=n_fft,
        )
        self.formatter = self._formatter(out_format)

    def forward(self, mc):
        """Convert mel-cepstrum to spectrum.

        Parameters
        ----------
        mc : Tensor [shape=(..., M+1)]
            Mel-cepstrum.

        Returns
        -------
        out : Tensor [shape=(..., L/2+1)]
            Spectrum.

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
        check_size(mc.size(-1), self.cep_order + 1, "dimension of cepstrum")
        return self._forward(self.mgc2c(mc), self.formatter)

    @staticmethod
    def _forward(c, formatter):
        sp = torch.fft.rfft(c, n=(c.size(-1) - 1) * 2)
        sp = formatter(sp)
        return sp

    @staticmethod
    def _func(mc, fft_length, alpha, gamma, norm, mul, n_fft, out_format):
        c = MelGeneralizedCepstrumToMelGeneralizedCepstrum._func(
            mc,
            fft_length // 2,
            in_alpha=alpha,
            in_gamma=gamma,
            in_norm=norm,
            in_mul=mul,
            out_alpha=0,
            out_gamma=0,
            out_norm=False,
            out_mul=False,
            n_fft=n_fft,
        )
        formatter = MelGeneralizedCepstrumToSpectrum._formatter(out_format)
        return MelGeneralizedCepstrumToSpectrum._forward(c, formatter)

    @staticmethod
    def _formatter(out_format):
        if out_format in (0, "db"):
            c = 20 / math.log(10)
            return lambda x: x.real * c
        elif out_format in (1, "log-magnitude"):
            return lambda x: x.real
        elif out_format in (2, "magnitude"):
            return lambda x: torch.exp(x.real)
        elif out_format in (3, "power"):
            return lambda x: torch.exp(2 * x.real)
        elif out_format in (4, "cycle"):
            return lambda x: x.imag / torch.pi
        elif out_format in (5, "radian"):
            return lambda x: x.imag
        elif out_format in (6, "degree"):
            c = 180 / torch.pi
            return lambda x: x.imag * c
        elif out_format == "complex":
            return lambda x: torch.polar(torch.exp(x.real), x.imag)
        raise ValueError(f"out_format {out_format} is not supported.")
