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
from .acorr import Autocorrelation


class Yingram(nn.Module):
    """Pitch-related feature extraction module based on YIN.

    Parameters
    ----------
    frame_length : int >= 1
        Frame length, :math:`L`.

    sample_rate : int >= 1
        Sample rate in Hz.

    lag_min : int >= 1
        Minimum lag in points.

    lag_max : int < L
        Maximum lag in points.

    n_bin : int >= 1
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
        super().__init__()

        if lag_max is None:
            lag_max = frame_length - 1

        assert 1 <= sample_rate
        assert 1 <= lag_min <= lag_max < frame_length
        assert 1 <= n_bin

        self.frame_length = frame_length
        self.lag_max = lag_max
        self.acorr = Autocorrelation(frame_length, lag_max - 1)
        lags, lags_ceil, lags_floor, ramp = self._precompute(
            sample_rate, lag_min, lag_max, n_bin
        )
        self.register_buffer("lags", lags)
        self.register_buffer("lags_ceil", lags_ceil)
        self.register_buffer("lags_floor", lags_floor)
        self.register_buffer("ramp", ramp)

    def forward(self, x):
        """Compute YIN derivatives.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Framed waveform.

        Returns
        -------
        out : Tensor [shape=(..., M)]
            Yingram.

        Examples
        --------
        >>> x = diffsptk.nrand(22050)
        >>> frame = diffsptk.Frame(2048, 441)
        >>> yingram = diffsptk.Yingram(2048)
        >>> y = yingram(frame(x))
        >>> y.shape
        torch.Size([51, 1580])

        """
        check_size(x.size(-1), self.frame_length, "frame length")
        return self._forward(
            x,
            self.acorr,
            self.lag_max,
            self.lags,
            self.lags_ceil,
            self.lags_floor,
            self.ramp,
        )

    @staticmethod
    def _forward(x, acorr, lag_max, lags, lags_ceil, lags_floor, ramp):
        W = x.size(-1)
        x0 = F.pad(x, (1, 0))
        s = torch.cumsum(x0 * x0, dim=-1)
        term1 = (s[..., W - lag_max + 1 :]).flip(-1)
        term2 = s[..., W:] - s[..., :lag_max]
        term3 = -2 * acorr(x)

        # Compute Eq. (7).
        d = (term1 + term2 + term3)[..., 1:]

        # Compute Eq. (8).
        d = ramp * d / (torch.cumsum(d, dim=-1) + 1e-7)

        # Compute Yingram.
        d0 = F.pad(d, (1, 0), value=1)
        numer = (lags - lags_floor) * (d0[..., lags_ceil] - d0[..., lags_floor])
        denom = lags_ceil - lags_floor
        y = numer / denom + d0[..., lags_floor]
        return y

    @staticmethod
    def _func(x, sample_rate, lag_min, lag_max, n_bin):
        if lag_max is None:
            lag_max = x.size(-1) - 1
        const = Yingram._precompute(
            sample_rate, lag_min, lag_max, n_bin, dtype=x.dtype, device=x.device
        )
        return Yingram._forward(
            x, lambda x: Autocorrelation._func(x, lag_max - 1), lag_max, *const
        )

    @staticmethod
    def _precompute(sample_rate, lag_min, lag_max, n_bin, dtype=None, device=None):
        midi_min = int(np.ceil(Yingram.lag2midi(lag_max, sample_rate)))
        midi_max = int(Yingram.lag2midi(lag_min, sample_rate))
        lags = Yingram.midi2lag(
            torch.arange(
                midi_min, midi_max + 1, 1 / n_bin, dtype=torch.double, device=device
            ),
            sample_rate,
        )
        lags_ceil = lags.ceil().long()
        lags_floor = lags.floor().long()
        ramp = torch.arange(1, lag_max, device=device)
        return to(lags, dtype=dtype), lags_ceil, lags_floor, ramp

    @staticmethod
    def midi2lag(midi, sample_rate):
        return sample_rate / (440 * 2 ** ((midi - 69) / 12))

    @staticmethod
    def lag2midi(lag, sample_rate):
        return 12 * np.log2(sample_rate / (440 * lag)) + 69
