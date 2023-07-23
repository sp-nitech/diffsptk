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
import torch.nn.functional as F

from ..misc.utils import check_size
from .acorr import AutocorrelationAnalysis


def midi2lag(midi, sample_rate=16000):
    return sample_rate / (440 * 2 ** ((midi - 69) / 12))


def lag2midi(lag, sample_rate=16000):
    return 12 * np.log2(sample_rate / (440 * lag)) + 69


class Yingram(nn.Module):
    """Pitch-related feature extraction module based on YIN.

    Parameters
    ----------
    frame_length : int >= 1 [scalar]
        Frame length, :math:`L`.

    sample_rate : int >= 1 [scalar]
        Sample rate in Hz.

    lag_min : int >= 1 [scalar]
        Minimum lag in points.

    lag_max : int <= :math:`L` [scalar]
        Maximum lag in points.

    n_bin : int >= 1 [scalar]
        Number of bins of Yingram to represent a semitone range.

    References
    ----------
    .. [1] A. Cheveigne and H. Kawahara, "YIN, a fundamental frequency estimator for
           speech and music," *The Journal of the Acoustical Society of America*,
           vol. 111, 2002.

    .. [2] H. Choi et al., "Neural analysis and synthesis: Reconstructing speech from
           self-supervised representations," *arXiv:2110.14513*, 2021.

    """

    def __init__(
        self,
        frame_length,
        sample_rate=22050,
        lag_min=22,
        lag_max=None,
        n_bin=20,
    ):
        super(Yingram, self).__init__()

        if lag_max is None:
            lag_max = frame_length

        self.frame_length = frame_length
        self.tau_max = lag_max

        assert 1 <= sample_rate
        assert 1 <= lag_min <= lag_max <= frame_length
        assert 1 <= n_bin

        self.acorr = AutocorrelationAnalysis(
            lag_max - 1,
            frame_length,
        )

        midi_min = int(np.ceil(lag2midi(lag_max, sample_rate)))
        midi_max = int(lag2midi(lag_min, sample_rate))
        lags = midi2lag(torch.arange(midi_min, midi_max + 1, 1 / n_bin), sample_rate)
        lags_ceil = lags.ceil().long()
        lags_floor = lags.floor().long()
        self.register_buffer("lags", lags)
        self.register_buffer("lags_ceil", lags_ceil)
        self.register_buffer("lags_floor", lags_floor)
        self.register_buffer("ramp", torch.arange(1, lag_max))

    def forward(self, x):
        """Compute YIN derivatives.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Framed waveform.

        Returns
        -------
        y : Tensor [shape=(..., M)]
            Yingram.

        Examples
        --------
        >>> x = diffsptk.nrand(2047)
        >>> yingram = diffsptk.Yingram(x.size(-1))
        >>> y = yingram(x)
        >>> y.shape
        torch.Size([1580])

        """
        W = self.frame_length
        check_size(x.size(-1), W, "frame length")

        x0 = F.pad(x, (1, 0))
        s = torch.cumsum(x0 * x0, dim=-1)
        term1 = (s[..., W - self.tau_max + 1 :]).flip(-1)
        term2 = s[..., W:] - s[..., : self.tau_max]

        r = self.acorr(x)
        term3 = -2 * r

        # Compute Eq. (7).
        d = (term1 + term2 + term3)[..., 1:]

        # Compute Eq. (8).
        d = self.ramp * d / (torch.cumsum(d, dim=-1) + 1e-7)

        # Compute Yingram.
        d0 = F.pad(d, (1, 0), value=1)
        numer = (self.lags - self.lags_floor) * (
            d0[..., self.lags_ceil] - d0[..., self.lags_floor]
        )
        denom = self.lags_ceil - self.lags_floor
        y = numer / denom + d0[..., self.lags_floor]
        return y
