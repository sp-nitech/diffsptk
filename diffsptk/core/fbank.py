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


class MelFilterBankAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/fbank.html>`_
    for details.

    Parameters
    ----------
    n_channel : int >= 1 [scalar]
        Number of mel-filter banks, :math:`C`.

    fft_length : int >= 2 [scalar]
        Number of FFT bins, :math:`L`.

    sample_rate : int >= 1 [scalar]
        Sample rate in Hz.

    f_min : float >= 0 [scalar]
        Minimum frequency in Hz.

    f_max : float <= sample_rate // 2 [scalar]
        Maximum frequency in Hz.

    floor : float > 0 [scalar]
        Minimum mel-filter bank output in linear scale.

    out_format : ['y', 'yE', 'y,E']
        `y` is mel-filber bank outpus and `E` is energy. If this is `yE`, the two output
        tensors are concatenated and return the tensor instead of the tuple.

    """

    def __init__(
        self,
        n_channel,
        fft_length,
        sample_rate,
        f_min=0,
        f_max=None,
        floor=1e-5,
        use_power=False,
        out_format="y",
    ):
        super(MelFilterBankAnalysis, self).__init__()

        self.floor = floor
        self.use_power = use_power
        self.out_format = out_format

        if f_max is None:
            f_max = sample_rate / 2

        assert 1 <= n_channel
        assert 2 <= fft_length
        assert 1 <= sample_rate
        assert 0 <= f_min < f_max <= sample_rate / 2
        assert 0 < self.floor

        if out_format == 0 or out_format == "y":
            self.format_func = lambda y, E: y
        elif out_format == 1 or out_format == "yE":
            self.format_func = lambda y, E: torch.cat((y, E), dim=-1)
        elif out_format == 2 or out_format == "y,E":
            self.format_func = lambda y, E: (y, E)
        else:
            raise ValueError(f"out_format {out_format} is not supported")

        def hz_to_mel(x):
            return 1127 * np.log(x / 700 + 1)

        def mel_to_hz(x):
            return 700 * (np.exp(x / 1127) - 1)

        lower_bin_index = max(1, int(f_min / sample_rate * fft_length + 1.5))
        upper_bin_index = min(
            fft_length // 2, int(f_max / sample_rate * fft_length + 0.5)
        )

        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)

        seed = np.arange(1, n_channel + 2)
        center_frequencies = (mel_max - mel_min) / (n_channel + 1) * seed + mel_min
        self.center_frequencies = mel_to_hz(center_frequencies)

        seed = np.arange(lower_bin_index, upper_bin_index)
        mel = hz_to_mel(sample_rate * seed / fft_length)
        lower_channel_map = [np.argmax(0 < (m <= center_frequencies)) for m in mel]

        diff = center_frequencies - np.insert(center_frequencies[:-1], 0, mel_min)
        weights = np.zeros((fft_length // 2 + 1, n_channel))
        for i, k in enumerate(seed):
            m = lower_channel_map[i]
            w = (center_frequencies[max(0, m)] - mel[i]) / diff[max(0, m)]
            if 0 < m:
                weights[k, m - 1] += w
            if m < n_channel:
                weights[k, m] += 1 - w

        self.register_buffer("H", numpy_to_torch(weights))

    def forward(self, x):
        """Apply mel-filter banks to STFT.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            Power spectrum.

        Returns
        -------
        y : Tensor [shape=(..., C)]
            Mel-filter bank output.

        E : Tensor [shape=(..., 1)]
            Energy.

        Examples
        --------
        >>> x = diffsptk.ramp(19)
        >>> stft = diffsptk.STFT(frame_length=10, frame_period=10, fft_length=32)
        >>> fbank = diffsptk.MelFilterBankAnalysis(4, 32, 8000)
        >>> y = fbank(stft(x))
        >>> y
        tensor([[0.1214, 0.4825, 0.6072, 0.3589],
                [3.3640, 3.4518, 2.7717, 0.5088]])

        """
        y = x if self.use_power else torch.sqrt(x)
        y = torch.matmul(y, self.H)
        y = torch.log(torch.clip(y, min=self.floor))
        E = (2 * x[..., 1:-1]).sum(-1) + x[..., 0] + x[..., -1]
        E = torch.log(E / (2 * (x.size(-1) - 1))).unsqueeze(-1)
        return self.format_func(y, E)
