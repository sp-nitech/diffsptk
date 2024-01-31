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

from ..misc.utils import numpy_to_torch
from ..misc.utils import replicate1
from .fbank import MelFilterBankAnalysis
from .levdur import LevinsonDurbin
from .mgc2mgc import MelGeneralizedCepstrumToMelGeneralizedCepstrum


class PerceptualLinearPredictiveCoefficientsAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/plp.html>`_
    for details.

    Parameters
    ----------
    mfcc_order : int >= 1 [scalar]
        Order of MFCC, :math:`M`.

    n_channel : int >= 1 [scalar]
        Number of mel-filter banks, :math:`C`.

    fft_length : int >= 2 [scalar]
        Number of FFT bins, :math:`L`.

    sample_rate : int >= 1 [scalar]
        Sample rate in Hz.

    lifter : int >= 1 [scalar]
        Liftering coefficient.

    compression_factor : float > 0 [scalar]
        Amplitude compression factor.

    f_min : float >= 0 [scalar]
        Minimum frequency in Hz.

    f_max : float <= sample_rate // 2 [scalar]
        Maximum frequency in Hz.

    floor : float > 0 [scalar]
        Minimum mel-filter bank output in linear scale.

    out_format : ['y', 'yE', 'yc', 'ycE']
        `y` is MFCC, `c` is C0, and `E` is energy.

    n_fft : int >> :math:`M` [scalar]
        Number of FFT bins. Accurate conversion requires the large value.

    """

    def __init__(
        self,
        plp_order,
        n_channel,
        fft_length,
        sample_rate,
        lifter=1,
        compression_factor=0.33,
        out_format="y",
        n_fft=512,
        **fbank_kwargs,
    ):
        super(PerceptualLinearPredictiveCoefficientsAnalysis, self).__init__()

        self.plp_order = plp_order
        self.compression_factor = compression_factor

        assert 1 <= self.plp_order < n_channel
        assert 1 <= lifter
        assert 0 < self.compression_factor

        if out_format == 0 or out_format == "y":
            self.format_func = lambda y, c, E: y
        elif out_format == 1 or out_format == "yE":
            self.format_func = lambda y, c, E: torch.cat((y, E), dim=-1)
        elif out_format == 2 or out_format == "yc":
            self.format_func = lambda y, c, E: torch.cat((y, c), dim=-1)
        elif out_format == 3 or out_format == "ycE":
            self.format_func = lambda y, c, E: torch.cat((y, c, E), dim=-1)
        else:
            raise ValueError(f"out_format {out_format} is not supported")

        self.fbank = MelFilterBankAnalysis(
            n_channel,
            fft_length,
            sample_rate,
            use_power=True,
            out_format="y,E",
            **fbank_kwargs,
        )
        self.levdur = LevinsonDurbin(self.plp_order)
        self.lpc2c = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
            self.plp_order,
            self.plp_order,
            in_gamma=-1,
            in_norm=True,
            in_mul=True,
            n_fft=n_fft,
        )

        f = self.fbank.center_frequencies[:-1] ** 2
        e = (f / (f + 1.6e5)) ** 2 * (f + 1.44e6) / (f + 9.61e6)
        self.register_buffer("equal_loudness_curve", numpy_to_torch(e))

        m = np.arange(self.plp_order + 1)
        v = 1 + (lifter / 2) * np.sin((np.pi / lifter) * m)
        v[0] = 2
        self.register_buffer("liftering_vector", numpy_to_torch(v))

    def forward(self, x):
        """Compute PLP.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            Power spectrum.

        Returns
        -------
        y : Tensor [shape=(..., M)]
            PLP without C0.

        E : Tensor [shape=(..., 1)]
            Energy.

        c : Tensor [shape=(..., 1)]
            C0.

        Examples
        --------
        >>> x = diffsptk.ramp(19)
        >>> stft = diffsptk.STFT(frame_length=10, frame_period=10, fft_length=32)
        >>> plp = diffsptk.PLP(4, 8, 32, 8000)
        >>> y = plp(stft(x))
        >>> y
        tensor([[-0.2896, -0.2356, -0.0586, -0.0387],
                [ 0.4468, -0.5820,  0.0104, -0.0505]])

        """
        y, E = self.fbank(x)
        y = (torch.exp(y) * self.equal_loudness_curve) ** self.compression_factor
        y = replicate1(y)
        y = torch.fft.hfft(y, norm="forward")[..., : self.plp_order + 1].real
        y = self.levdur(y)
        y = self.lpc2c(y)
        y *= self.liftering_vector
        c, y = torch.split(y, [1, self.plp_order], dim=-1)
        return self.format_func(y, c, E)
