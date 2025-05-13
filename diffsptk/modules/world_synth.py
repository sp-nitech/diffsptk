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

# ----------------------------------------------------------------- #
# Copyright (c) 2010  M. Morise                                     #
#                                                                   #
# All rights reserved.                                              #
#                                                                   #
# Redistribution and use in source and binary forms, with or        #
# without modification, are permitted provided that the following   #
# conditions are met:                                               #
#                                                                   #
# - Redistributions of source code must retain the above copyright  #
#   notice, this list of conditions and the following disclaimer.   #
# - Redistributions in binary form must reproduce the above         #
#   copyright notice, this list of conditions and the following     #
#   disclaimer in the documentation and/or other materials provided #
#   with the distribution.                                          #
# - Neither the name of the M. Morise nor the names of its          #
#   contributors may be used to endorse or promote products derived #
#   from this software without specific prior written permission.   #
#                                                                   #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND            #
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,       #
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF          #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE          #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS #
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,          #
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED   #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,     #
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON #
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,   #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY    #
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           #
# POSSIBILITY OF SUCH DAMAGE.                                       #
# ----------------------------------------------------------------- #

import torch

from ..third_party.world import get_minimum_phase_spectrum, interp1
from ..utils.private import TAU
from .base import BaseNonFunctionalModule


class WorldSynthesis(BaseNonFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/world_synth.html>`_
    for details. Note that the gradients do not propagated through F0.

    Parameters
    ----------
    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    sample_rate : int >= 8000
        The sample rate in Hz.

    fft_length : int >= 1024
        The number of FFT bins, :math:`L`.

    default_f0 : float > 0
        The F0 value used when the input F0 is unvoiced.

    """

    def __init__(
        self,
        frame_period: int,
        sample_rate: int,
        fft_length: int,
        *,
        default_f0: float = 500,
    ) -> None:
        super().__init__()

        if frame_period <= 0:
            raise ValueError("frame_period must be positive.")
        if sample_rate < 8000:
            raise ValueError("sample_rate must be at least 8000 Hz.")
        if fft_length < 1024:
            raise ValueError("fft_length must be at least 1024.")

        self.frame_period = frame_period
        self.sample_rate = sample_rate
        self.fft_length = fft_length
        self.default_f0 = default_f0

        self.register_buffer("ramp", torch.arange(fft_length))

        # GetDCRemover()
        ramp = self.ramp[1 : fft_length // 2 + 1]
        dc_remover = 0.5 - 0.5 * torch.cos(TAU / (1 + fft_length) * ramp)
        dc_component = 2 * torch.sum(dc_remover)
        dc_remover /= dc_component
        dc_remover = torch.cat([dc_remover, dc_remover.flip(-1)], dim=-1)
        self.register_buffer("dc_remover", dc_remover)

    def forward(
        self, f0: torch.Tensor, ap: torch.Tensor, sp: torch.Tensor
    ) -> torch.Tensor:
        """Synthesize speech using WORLD vocoder.

        Parameters
        ----------
        f0 : Tensor [shape=(B, T/P) or (T/P,)]
            The F0 in Hz.

        ap : Tensor [shape=(B, T/P, L/2+1) or (T/P, L/2+1)]
            The aperiodicity in [0, 1].

        sp : Tensor [shape=(B, T/P, L/2+1) or (T/P, L/2+1)]
            The spectral envelope (power spectrum).

        Returns
        -------
        out : Tensor [shape=(B, T) or (T,)]
            The synthesized speech waveform.

        Examples
        --------
        >>> x = diffsptk.sin(1000, 80)
        >>> pitch = diffsptk.Pitch(160, 8000, out_format="f0")
        >>> f0 = pitch(x)
        >>> aperiodicity = diffsptk.Aperiodicity(160, 16000, 1024)
        >>> ap = aperiodicity(x, f0)
        >>> pitch_spec = diffsptk.PitchAdaptiveSpectralAnalysis(160, 8000, 1024)
        >>> sp = pitch_spec(x, f0)
        >>> world_synth = diffsptk.WorldSynthesis(160, 8000, 1024)
        >>> y = world_synth(f0, ap, sp)
        >>> y.shape
        torch.Size([1120])

        """
        is_batched_input = f0.ndim == 2
        if not is_batched_input:
            f0 = f0.unsqueeze(0)
            ap = ap.unsqueeze(0)
            sp = sp.unsqueeze(0)

        # Check the input shape.
        if f0.dim() != 2:
            raise ValueError("f0 must be 1D or 2D tensor.")
        if ap.dim() != 3 or sp.dim() != 3:
            raise ValueError("ap and sp must be 2D or 3D tensor.")
        if len(set([f0.shape[0], ap.shape[0], sp.shape[0]])) != 1:
            raise ValueError("f0, ap, and sp must have the same batch size.")
        if len(set([f0.shape[1], ap.shape[1], sp.shape[1]])) != 1:
            raise ValueError("f0, ap, and sp must have the same length.")
        if len(set([ap.shape[2], sp.shape[2]])) != 1:
            raise ValueError("ap and sp must have the same dimension.")

        # Get the input shape.
        B, N, D = sp.shape
        T = N * self.frame_period

        # Restrict the input range.
        eps = 1e-6
        ap = torch.clip(ap, min=eps, max=1 - eps)
        sp = torch.clip(sp, min=eps)

        # GetTemporalParametersForTimeBase()
        f_min = self.sample_rate / self.fft_length + 1
        coarse_f0 = torch.where(f0 < f_min, 0, f0).detach()
        coarse_vuv = (0 < coarse_f0).type(coarse_f0.dtype)
        time_axis = (
            torch.arange(f0.shape[-1] * self.frame_period, device=f0.device)
            / self.sample_rate
        )
        time_axis = time_axis.repeat(B, 1)
        coarse_time_axis = torch.arange(
            coarse_f0.shape[-1], device=coarse_f0.device
        ) * (self.frame_period / self.sample_rate)
        coarse_time_axis = coarse_time_axis.repeat(B, 1)
        interpolated_f0 = interp1(
            coarse_time_axis, coarse_f0, time_axis, batching=(True, True)
        )
        interpolated_vuv = interp1(
            coarse_time_axis, coarse_vuv, time_axis, batching=(True, True)
        )
        interpolated_vuv = 0.5 < interpolated_vuv
        interpolated_f0 = torch.where(
            interpolated_vuv, interpolated_f0, self.default_f0
        )

        # GetPulseLocationsForTimeBase()
        total_phase = torch.cumsum(
            TAU / self.sample_rate * interpolated_f0.double(), dim=-1
        ).type(f0.dtype)
        wrap_phase = torch.fmod(total_phase, TAU)
        wrap_phase_abs = torch.abs(torch.diff(wrap_phase))
        pulse_locations_index = torch.nonzero(torch.pi < wrap_phase_abs, as_tuple=True)
        pulse_locations = time_axis[pulse_locations_index]
        vuv = interpolated_vuv[pulse_locations_index].unsqueeze(-1)
        batch_index, time_index = pulse_locations_index
        y1 = wrap_phase[pulse_locations_index] - TAU
        y2 = wrap_phase[batch_index, time_index + 1]
        pulse_locations_time_shift = -y1 / (y2 - y1) / self.sample_rate

        # GetSpectralEnvelope()
        frame = pulse_locations * (self.sample_rate / self.frame_period)
        frame_floor = frame.floor().long().clip(max=N - 1)
        frame_ceil = frame.ceil().long().clip(max=N - 1)
        interpolation = (frame - frame_floor).unsqueeze(-1)
        lower_weight = 1 - interpolation
        upper_weight = interpolation
        spectral_envelope = (
            lower_weight * sp[batch_index, frame_floor]
            + upper_weight * sp[batch_index, frame_ceil]
        )

        # GetAperiodicRatio()
        aperiodic_ratio = (
            lower_weight * ap[batch_index, frame_floor]
            + upper_weight * ap[batch_index, frame_ceil]
        ) ** 2

        # GetPeriodicResponse()
        weight = 1 - aperiodic_ratio
        spectrum = get_minimum_phase_spectrum(weight * spectral_envelope)

        # GetSpectrumWithFractionalTimeShift()
        coefficient = (
            TAU * self.sample_rate / self.fft_length * pulse_locations_time_shift
        )
        phase = torch.exp(-1j * self.ramp[:D] * coefficient.unsqueeze(-1))
        periodic_response = torch.fft.hfft(spectrum * phase)
        periodic_response = torch.cat(
            [periodic_response[..., :1], periodic_response[..., 1:].flip(-1)], dim=-1
        )
        periodic_response = torch.fft.fftshift(periodic_response, dim=-1)

        # RemoveDCComponent()
        H = self.fft_length // 2
        dc_component = periodic_response[..., H:].sum(-1, keepdim=True)
        dd = -dc_component * self.dc_remover
        periodic_response = torch.cat(
            (dd[..., :H], periodic_response[..., H:] + dd[..., H:]), dim=-1
        )
        periodic_response = periodic_response * (0.5 < vuv)

        # GetNoiseSpectrum()
        noise_size = torch.diff(time_index, append=time_index[-1:])
        noise_size = noise_size.clip(min=0).unsqueeze(-1)
        noise_waveform = torch.randn_like(periodic_response)
        mask = self.ramp < noise_size
        noise_waveform = noise_waveform * mask
        average = noise_waveform.sum(dim=-1, keepdim=True) / noise_size
        average = torch.nan_to_num(average)
        noise_waveform = (noise_waveform - average) * mask
        noise_spectrum = torch.fft.rfft(noise_waveform)

        # GetAperiodicResponse()
        weight = torch.where(0 < vuv, aperiodic_ratio, 1)
        spectrum = (
            get_minimum_phase_spectrum(weight * spectral_envelope) * noise_spectrum
        )
        aperiodic_response = torch.fft.hfft(spectrum)
        aperiodic_response = torch.cat(
            [aperiodic_response[..., :1], aperiodic_response[..., 1:].flip(-1)], dim=-1
        )
        aperiodic_response = torch.fft.fftshift(aperiodic_response, dim=-1)

        # Synthesis()
        sqrt_noise_size = torch.sqrt(noise_size)
        response = (
            periodic_response * sqrt_noise_size + aperiodic_response
        ) / self.fft_length
        margin = (
            (self.fft_length + self.frame_period - 1)
            // self.frame_period
            * self.frame_period
        )
        T_ = T + margin
        index = (batch_index * T_ + time_index).unsqueeze(-1) + self.ramp
        y = torch.zeros((B, T_), device=sp.device)
        y.view(-1).scatter_add_(
            dim=-1,
            index=index.view(-1),
            src=response.view(-1),
        )
        y = torch.narrow(y, dim=-1, start=H, length=T)

        if not is_batched_input:
            y = y.squeeze(0)
        return y
