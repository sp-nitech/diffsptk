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
import torch.nn.functional as F

from ...modules.frame import Frame
from ...utils.private import cexp


def dc_correction(
    power_spectrum: torch.Tensor,
    f0: torch.Tensor,
    sample_rate: int,
    fft_length: int,
    ramp: torch.Tensor,
) -> torch.Tensor:
    rate = sample_rate / fft_length
    low_frequency_axis = ramp[: fft_length // 2 + 1] * rate
    corrected_power_spectrum = interp1Q(f0, -rate, power_spectrum, low_frequency_axis)
    mask = low_frequency_axis < f0
    power_spectrum = power_spectrum + corrected_power_spectrum * mask
    return power_spectrum


def get_minimum_phase_spectrum(spectrum: torch.Tensor) -> torch.Tensor:
    one_sided_fft_length = spectrum.shape[-1]
    cepstrum = torch.fft.irfft(0.5 * torch.log(spectrum))
    cepstrum = torch.cat(
        (
            cepstrum[..., :1],
            2 * cepstrum[..., 1 : one_sided_fft_length - 1],
            cepstrum[..., one_sided_fft_length - 1 : one_sided_fft_length],
        ),
        dim=-1,
    )
    return cexp(torch.fft.rfft(cepstrum, n=2 * (one_sided_fft_length - 1)))


def get_windowed_waveform(
    x: torch.Tensor,
    f0: torch.Tensor,
    window_length_ratio: float,
    bias_ratio: float,
    frame_period: int,
    sample_rate: int,
    fft_length: int,
    window_type: str,
    normalize_window: bool,
    eps: float,
    ramp: torch.Tensor,
) -> torch.Tensor:
    # SetParametersForGetWindowedWaveform()
    half_window_length = torch.round(window_length_ratio / 2 * sample_rate / f0).long()
    bias = torch.round(bias_ratio * sample_rate / f0).long()
    base_index = ramp[:fft_length] - bias - fft_length // 2
    position = base_index / (window_length_ratio / 2 * sample_rate)
    z = torch.pi * position * f0
    # https://github.com/mmorise/World/issues/150
    if window_type == "hanning":
        window = 0.5 + 0.5 * torch.cos(z)
    elif window_type == "blackman":
        window = 0.42 + 0.5 * torch.cos(z) + 0.08 * torch.cos(2 * z)
    else:
        raise RuntimeError
    mask1 = -half_window_length <= base_index
    mask2 = base_index <= half_window_length
    mask = torch.logical_and(mask1, mask2)
    window *= mask
    if normalize_window:
        window = window / torch.linalg.vector_norm(window, dim=-1, keepdim=True)

    # GetWindowedWaveform()
    waveform = (
        Frame._func(
            x,
            fft_length,
            frame_period,
            center=True,
            zmean=False,
            mode="replicate",
        )
        * window
    )
    waveform += torch.randn_like(waveform) * eps * mask
    tmp_weight1 = waveform.sum(dim=-1, keepdim=True)
    tmp_weight2 = window.sum(dim=-1, keepdim=True)
    weighting_coefficient = tmp_weight1 / tmp_weight2
    waveform -= window * weighting_coefficient
    return waveform


# From https://github.com/pytorch/pytorch/issues/50334#issuecomment-2304751532
def interp1(
    x: torch.Tensor,
    y: torch.Tensor,
    xq: torch.Tensor,
    method: str = "linear",
    batching: tuple = (False, False),
):
    if not batching[0]:
        x = x.repeat(*xq.shape[0:-1], 1)
    if not batching[1]:
        y = y.repeat(*xq.shape[0:-1], 1)
    m = torch.diff(y) / torch.diff(x)
    b = y[..., :-1] - m * x[..., :-1]
    indices = torch.searchsorted(x, xq, right=False)
    if method == "linear":
        m = F.pad(m, (1, 1))
        b = torch.cat([y[..., :1], b, y[..., -1:]], dim=-1)
    elif method == "*linear":
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)
    else:
        raise ValueError(f"Unknown method: {method}")
    values = m.gather(-1, indices) * xq + b.gather(-1, indices)
    return values


def interp1Q(
    x: torch.Tensor, shift: float, y: torch.Tensor, xi: torch.Tensor
) -> torch.Tensor:
    z = (xi - x) / shift
    xi_base = torch.clip(z.long(), min=0)
    xi_fraction = z - xi_base
    delta_y = torch.diff(y, dim=-1, append=y[..., -1:])
    yi = torch.gather(y, -1, xi_base) + torch.gather(delta_y, -1, xi_base) * xi_fraction
    return yi


def linear_smoothing(
    power_spectrum: torch.Tensor,
    width: torch.Tensor,
    sample_rate: int,
    fft_length: int,
    ramp: torch.Tensor,
) -> torch.Tensor:
    one_sided_length = fft_length // 2 + 1
    rate = sample_rate / fft_length
    boundary = (width / rate).long() + 1
    max_boundary = torch.amax(boundary)
    mirroring_spectrum = F.pad(
        power_spectrum, (max_boundary, max_boundary), mode="reflect"
    )
    bias = max_boundary - boundary
    mask = bias <= ramp[:max_boundary]
    mask = F.pad(mask, (0, one_sided_length + max_boundary), value=True)
    mirroring_spectrum = mirroring_spectrum * mask
    mirroring_segment = torch.cumsum(mirroring_spectrum * rate, dim=-1)
    origin_of_mirroring_axis = -(max_boundary - 0.5) * rate
    frequency_axis = ramp[:one_sided_length] * rate - width / 2
    low_levels = interp1Q(
        origin_of_mirroring_axis, rate, mirroring_segment, frequency_axis
    )
    high_levels = interp1Q(
        origin_of_mirroring_axis, rate, mirroring_segment, frequency_axis + width
    )
    power_spectrum = (high_levels - low_levels) / width
    return power_spectrum
