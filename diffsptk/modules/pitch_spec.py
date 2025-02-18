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
from torch import nn

from ..misc.world import dc_correction
from ..misc.world import get_windowed_waveform
from ..misc.world import linear_smoothing
from .spec import Spectrum


class PitchAdaptiveSpectralAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/pitch_spec.html>`_
    for details.

    Parameters
    ----------
    frame_period : int >= 1
        Frame period in sample, :math:`P`.

    sample_rate : int >= 8000
        Sample rate in Hz.

    fft_length : int >= 1024
        Number of FFT bins, :math:`L`.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        Output format.

    q1 : float
        A parameter used for spectral recovery.

    default_f0 : float > 0
        F0 value used when the input F0 is unvoiced.

    References
    ----------
    .. [1] M. Morise, "CheapTrick, a spectral envelope estimator for high-quality speech
           synthesis", *Speech Communication*, vol. 67, pp. 1-7, 2015.

    """

    def __init__(
        self,
        frame_period,
        sample_rate,
        fft_length,
        out_format="power",
        q1=-0.15,
        default_f0=500,
    ):
        super().__init__()

        assert 1 <= frame_period
        assert 8000 <= sample_rate

        self.frame_period = frame_period
        self.sample_rate = sample_rate
        self.fft_length = fft_length
        self.formatter = self._formatter(out_format)

        # GetF0FloorForCheapTrick()
        self.f_min = 3 * sample_rate / (fft_length - 3)
        assert self.f_min <= default_f0

        # GetFFTSizeForCheapTrick()
        min_fft_length = 2 ** (
            1 + int(np.log(3 * sample_rate / self.f_min + 1) / np.log(2))
        )
        assert min_fft_length <= fft_length

        # Set WORLD constants.
        self.q1 = q1
        self.default_f0 = default_f0

        # Prepare modules.
        self.spec = Spectrum(fft_length)

        self.register_buffer("ramp", torch.arange(fft_length))

    def forward(self, x, f0):
        """Estimate spectral envelope.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Waveform.

        f0 : Tensor [shape=(..., T/P)]
            F0 in Hz.

        Returns
        -------
        out : Tensor [shape=(..., T/P, L/2+1)]
            Spectral envelope.

        Examples
        --------
        >>> x = diffsptk.sin(1000, 80)
        >>> pitch = diffsptk.Pitch(160, 8000, out_format="f0")
        >>> f0 = pitch(x)
        >>> f0.shape
        torch.Size([7])
        >>> pitch_spec = diffsptk.PitchAdaptiveSpectralAnalysis(160, 8000, 1024)
        >>> sp = pitch_spec(x, f0)
        >>> sp.shape
        torch.Size([7, 513])

        """
        f0 = torch.where(f0 <= self.f_min, self.default_f0, f0).unsqueeze(-1).detach()

        # GetWindowedWaveform()
        waveform = get_windowed_waveform(
            x,
            f0,
            3,
            0,
            self.frame_period,
            self.sample_rate,
            self.fft_length,
            "hanning",
            True,
            1e-12,
            self.ramp,
        )

        # GetPowerSpectrum()
        power_spectrum = self.spec(waveform)

        # DCCorrection()
        power_spectrum = dc_correction(
            power_spectrum, f0, self.sample_rate, self.fft_length, self.ramp
        )

        # LinearSmoothing()
        power_spectrum = linear_smoothing(
            power_spectrum, f0 * (2 / 3), self.sample_rate, self.fft_length, self.ramp
        )

        # AddInfinitesimalNoise()
        power_spectrum += (
            torch.randn_like(power_spectrum).abs() * torch.finfo(x.dtype).eps
        )

        # SmoothingWithRecovery()
        one_sided_length = self.fft_length // 2 + 1
        quefrency = self.ramp[:one_sided_length] / self.sample_rate
        z = torch.pi * f0 * quefrency
        smoothing_lifter = torch.sin(z) / z
        compensation_lifter = (1 - 2 * self.q1) + 2 * self.q1 * torch.cos(2 * z)
        smoothing_lifter[..., 0] = 1
        cepstrum = torch.fft.irfft(torch.log(power_spectrum))[..., :one_sided_length]
        log_power_spectrum = torch.fft.hfft(
            cepstrum * smoothing_lifter * compensation_lifter,
        )[..., :one_sided_length]

        spectral_envelope = self.formatter(log_power_spectrum)
        return spectral_envelope

    @staticmethod
    def _formatter(out_format):
        if out_format in (0, "db"):
            return lambda x: x * (10 / np.log(10))
        elif out_format in (1, "log-magnitude"):
            return lambda x: x / 2
        elif out_format in (2, "magnitude"):
            return lambda x: torch.exp(x / 2)
        elif out_format in (3, "power"):
            return lambda x: torch.exp(x)
        raise ValueError(f"out_format {out_format} is not supported.")
