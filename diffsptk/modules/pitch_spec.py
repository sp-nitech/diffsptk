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
import torch.nn.functional as F
from torch import nn

from .frame import Frame
from .spec import Spectrum


class PitchAdaptiveSpectralAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/pitch_spec.html>`_
    for details.

    Parameters
    ----------
    frame_period : int >= 1
        Frame period in sample, :math:`P`.

    fft_length : int >= 2
        Number of FFT bins, :math:`L`.

    sample_rate : int >= 8000
        Sample rate in Hz.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        Output format.

    q1 : float
        A parameter used for spectral recovery.

    default_f0 : float > 0
        F0 value used when the input F0 is unvoiced.

    safe_min : float > 0
        A small value added to random values to avoid computation errors.

    """

    def __init__(
        self,
        frame_period,
        fft_length,
        sample_rate,
        out_format="power",
        q1=-0.15,
        default_f0=500,
    ):
        super().__init__()

        assert 1 <= frame_period
        assert 8000 <= sample_rate

        self.frame_period = frame_period
        self.fft_length = fft_length
        self.sample_rate = sample_rate
        self.formatter = self._formatter(out_format)

        # GetF0FloorForCheapTrick()
        self.f_min = 3 * sample_rate / (fft_length - 3)

        # GetFFTSizeForCheapTrick
        min_fft_length = 2 ** (
            1 + int(np.log(3 * sample_rate / self.f_min + 1) / np.log(2))
        )
        assert min_fft_length <= fft_length

        # Set WORLD constants.
        self.q1 = q1
        self.default_f0 = default_f0

        # Prepare modules.
        self.frame = Frame(fft_length, frame_period, mode="replicate")
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
        >>> pitch_spec = diffsptk.PitchAdaptiveSpectralAnalysis(160, 1024, 8000)
        >>> sp = pitch_spec(x, f0)
        >>> sp.shape
        torch.Size([7, 513])

        """
        rate = self.sample_rate / self.fft_length

        # SetParametersForGetWindowedWaveform()
        f0 = torch.where(f0 <= self.f_min, self.default_f0, f0).unsqueeze(-1).detach()
        half_window_length = torch.round(1.5 * self.sample_rate / f0).long()
        half_fft_length = self.fft_length // 2
        base_index = self.ramp - half_fft_length
        position = base_index / 1.5 / self.sample_rate
        window = 0.5 * torch.cos(torch.pi * position * f0) + 0.5
        mask1 = -half_window_length <= base_index
        mask2 = base_index <= half_window_length
        mask = torch.logical_and(mask1, mask2)
        window *= mask
        window = window / torch.linalg.vector_norm(window, dim=-1, keepdim=True)

        # GetWindowedWaveform()
        waveform = self.frame(x) * window
        waveform += torch.randn_like(waveform) * 1e-12 * mask
        tmp_weight1 = waveform.sum(dim=-1, keepdim=True)
        tmp_weight2 = window.sum(dim=-1, keepdim=True)
        weighting_coefficient = tmp_weight1 / tmp_weight2
        waveform -= window * weighting_coefficient

        # GetPowerSpectrum()
        power_spectrum = self.spec(waveform)

        def interp1Q(x, shift, y, xi):
            z = (xi - x) / shift
            xi_base = torch.clip(z.long(), min=0)
            xi_fraction = z - xi_base
            delta_y = torch.diff(y, dim=-1, append=y[..., -1:])
            yi = (
                torch.gather(y, -1, xi_base)
                + torch.gather(delta_y, -1, xi_base) * xi_fraction
            )
            return yi

        # DCCorrection()
        one_sided_length = half_fft_length + 1
        low_frequency_axis = self.ramp[:one_sided_length] * rate
        corrected_power_spectrum = interp1Q(
            f0, -rate, power_spectrum, low_frequency_axis
        )
        mask = low_frequency_axis < f0
        power_spectrum = power_spectrum + corrected_power_spectrum * mask

        # LinearSmoothing()
        width = f0 * 2 / 3
        boundary = (width / rate).long() + 1
        max_boundary = torch.amax(boundary)
        mirroring_spectrum = F.pad(
            power_spectrum, (max_boundary, max_boundary), mode="reflect"
        )
        bias = max_boundary - boundary
        mask = bias <= self.ramp[:max_boundary]
        mask = F.pad(mask, (0, one_sided_length + max_boundary), value=True)
        mirroring_spectrum = mirroring_spectrum * mask
        mirroring_segment = torch.cumsum(mirroring_spectrum * rate, dim=-1)
        origin_of_mirroring_axis = -(max_boundary - 0.5) * rate
        frequency_axis = self.ramp[:one_sided_length] * rate - width / 2
        low_levels = interp1Q(
            origin_of_mirroring_axis, rate, mirroring_segment, frequency_axis
        )
        high_levels = interp1Q(
            origin_of_mirroring_axis, rate, mirroring_segment, frequency_axis + width
        )
        power_spectrum = (high_levels - low_levels) / width

        # AddInfinitesimalNoise()
        power_spectrum += (
            torch.randn_like(power_spectrum).abs() * torch.finfo(x.dtype).eps
        )

        # SmoothingWithRecovery()
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
