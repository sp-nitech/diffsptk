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

from ..misc.utils import check_size
from ..misc.utils import get_values
from ..misc.utils import to
from .acorr import Autocorrelation
from .base import BaseFunctionalModule


class Yingram(BaseFunctionalModule):
    """Pitch-related feature extraction module based on YIN.

    Parameters
    ----------
    frame_length : int >= 1
        The frame length in samples, :math:`L`.

    sample_rate : int >= 8000
        The sample rate in Hz.

    lag_min : int >= 1
        The minimum lag in points.

    lag_max : int < L
        The maximum lag in points.

    n_bin : int >= 1
        The number of bins to represent a semitone range.

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

        self.in_dim = frame_length

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("lags", tensors[0])
        self.register_buffer("lags_ceil", tensors[1])
        self.register_buffer("lags_floor", tensors[2])
        self.register_buffer("ramp", tensors[3])

    def forward(self, x):
        """Compute the YIN derivatives from the waveform.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            The framed waveform.

        Returns
        -------
        out : Tensor [shape=(..., M)]
            The Yingram.

        Examples
        --------
        >>> x = diffsptk.nrand(22050)
        >>> frame = diffsptk.Frame(2048, 441)
        >>> yingram = diffsptk.Yingram(2048)
        >>> y = yingram(frame(x))
        >>> y.shape
        torch.Size([51, 1580])

        """
        check_size(x.size(-1), self.in_dim, "frame length")
        return self._forward(x, **self._buffers)

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(frame_length, sample_rate, lag_min, lag_max, n_bin):
        if sample_rate < 8000:
            raise ValueError("sample_rate must be greater than or equal to 8000.")
        if not (1 <= lag_min <= lag_max < frame_length):
            raise ValueError("Invalid lag_min and lag_max.")
        if n_bin <= 0:
            raise ValueError("n_bin must be positive.")

    @staticmethod
    def _func(x, *args, **kwargs):
        _, _, tensors = Yingram._precompute(
            x.size(-1), *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return Yingram._forward(x, *tensors)

    @staticmethod
    def _precompute(
        frame_length, sample_rate, lag_min, lag_max, n_bin, device=None, dtype=None
    ):
        if lag_max is None:
            lag_max = frame_length - 1
        midi_min = int(np.ceil(Yingram.lag2midi(lag_max, sample_rate)))
        midi_max = int(Yingram.lag2midi(lag_min, sample_rate))
        lags = Yingram.midi2lag(
            torch.arange(
                midi_min, midi_max + 1, 1 / n_bin, device=device, dtype=torch.double
            ),
            sample_rate,
        )
        lags_ceil = lags.ceil().long()
        lags_floor = lags.floor().long()
        ramp = torch.arange(1, lag_max, device=device)
        return None, None, (to(lags, dtype=dtype), lags_ceil, lags_floor, ramp)

    @staticmethod
    def _forward(x, lags, lags_ceil, lags_floor, ramp):
        lag_max = len(ramp) + 1
        W = x.size(-1)
        x0 = F.pad(x, (1, 0))
        s = torch.cumsum(x0 * x0, dim=-1)
        term1 = (s[..., W - lag_max + 1 :]).flip(-1)
        term2 = s[..., W:] - s[..., :lag_max]
        term3 = -2 * Autocorrelation._func(x, lag_max - 1)

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
    def midi2lag(midi, sample_rate):
        return sample_rate / (440 * 2 ** ((midi - 69) / 12))

    @staticmethod
    def lag2midi(lag, sample_rate):
        return 12 * np.log2(sample_rate / (440 * lag)) + 69
