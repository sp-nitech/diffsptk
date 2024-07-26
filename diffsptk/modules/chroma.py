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
import torch.nn.functional as F

from ..misc.utils import check_size
from ..misc.utils import to


class ChromaFilterBankAnalysis(nn.Module):
    """Chroma-filter bank analysis.

    Parameters
    ----------
    n_channel : int >= 1
        Number of chroma-filter banks, :math:`C`.

    fft_length : int >= 2
        Number of FFT bins, :math:`L`.

    sample_rate : int >= 1
        Sample rate in Hz.

    norm : float
        Normalization factor.

    use_power : bool
        If True, use power spectrum instead of amplitude spectrum.

    """

    def __init__(
        self,
        n_channel,
        fft_length,
        sample_rate,
        norm=float("inf"),
        use_power=True,
    ):
        super().__init__()

        assert 1 <= n_channel
        assert 2 <= fft_length
        assert 1 <= sample_rate

        self.fft_length = fft_length
        self.norm = norm
        self.use_power = use_power

        self.register_buffer("H", self._precompute(n_channel, fft_length, sample_rate))

    def forward(self, x):
        """Apply chroma-filter banks to STFT.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            Power spectrum.

        Returns
        -------
        out : Tensor [shape=(..., C)]
            Chroma-filter bank output.

        Examples
        --------
        >>> sr = 16000
        >>> x = diffsptk.sin(500, period=sr/440)
        >>> stft = diffsptk.STFT(frame_length=512, frame_period=512, fft_length=512)
        >>> chroma = diffsptk.ChromaFilterBankAnalysis(12, 512, sr)
        >>> y = chroma(stft(x))
        >>> y
        tensor([[0.1146, 0.0574, 0.0457, 0.0468, 0.0519, 0.0789, 0.1736, 0.4285, 0.7416,
                 1.0000, 0.7806, 0.3505]])

        """
        check_size(x.size(-1), self.fft_length // 2 + 1, "dimension of spectrum")
        return self._forward(x, self.norm, self.use_power, self.H)

    @staticmethod
    def _forward(x, norm, use_power, H):
        y = x if use_power else torch.sqrt(x)
        y = torch.matmul(y, H)
        y = F.normalize(y, p=norm, dim=-1)
        return y

    @staticmethod
    def _func(
        x,
        n_channel,
        sample_rate,
        norm,
        use_power,
    ):
        H = ChromaFilterBankAnalysis._precompute(
            n_channel,
            2 * (x.size(-1) - 1),
            sample_rate,
            dtype=x.dtype,
            device=x.device,
        )
        return ChromaFilterBankAnalysis._forward(x, norm, use_power, H)

    @staticmethod
    def _precompute(n_channel, fft_length, sample_rate, dtype=None, device=None):
        import librosa

        weights = librosa.filters.chroma(
            sr=sample_rate,
            n_fft=fft_length,
            n_chroma=n_channel,
            base_c=True,
            dtype=np.float64,
        ).T
        weights = torch.from_numpy(weights)
        return to(weights, dtype=dtype, device=device)
