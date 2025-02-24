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
import torch.nn.functional as F

from ..modules.frame import Frame


def dc_correction(power_spectrum, f0, sample_rate, fft_length, ramp):
    rate = sample_rate / fft_length
    low_frequency_axis = ramp[: fft_length // 2 + 1] * rate
    corrected_power_spectrum = interp1Q(f0, -rate, power_spectrum, low_frequency_axis)
    mask = low_frequency_axis < f0
    power_spectrum = power_spectrum + corrected_power_spectrum * mask
    return power_spectrum


def get_windowed_waveform(
    x,
    f0,
    window_length_ratio,
    bias_ratio,
    frame_period,
    sample_rate,
    fft_length,
    window_type,
    normalize_window,
    eps,
    ramp,
):
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


def interp1Q(x, shift, y, xi):
    z = (xi - x) / shift
    xi_base = torch.clip(z.long(), min=0)
    xi_fraction = z - xi_base
    delta_y = torch.diff(y, dim=-1, append=y[..., -1:])
    yi = torch.gather(y, -1, xi_base) + torch.gather(delta_y, -1, xi_base) * xi_fraction
    return yi


def linear_smoothing(power_spectrum, width, sample_rate, fft_length, ramp):
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
