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
from torch import nn

from ..misc.utils import check_size
from ..misc.utils import to


class Window(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/window.html>`_
    for details.

    Parameters
    ----------
    in_length : int >= 1
        Input length or window length, :math:`L_1`.

    out_length : int >= L1 or None
        Output length, :math:`L_2`. If :math:`L_2 > L_1`, output is zero-padded.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular']
        Window type.

    norm : ['none', 'power', 'magnitude']
        Normalization type of window.

    """

    def __init__(self, in_length, out_length=None, *, window="blackman", norm="power"):
        super().__init__()

        assert 1 <= in_length

        self.in_length = in_length
        self.out_length = out_length
        self.register_buffer("window", self._precompute(self.in_length, window, norm))

    def forward(self, x):
        """Apply a window function to given waveform.

        Parameters
        ----------
        x : Tensor [shape=(..., L1)]
            Framed waveform.

        Returns
        -------
        out : Tensor [shape=(..., L2)]
            Windowed waveform.

        Examples
        --------
        >>> x = torch.ones(5)
        >>> window = diffsptk.Window(5, out_length=7, window="hamming", norm="none")
        >>> y = window(x)
        >>> y
        tensor([0.0800, 0.5400, 1.0000, 0.5400, 0.0800, 0.0000, 0.0000])

        """
        check_size(x.size(-1), self.in_length, "input length")
        return self._forward(x, self.out_length, self.window)

    @staticmethod
    def _forward(x, out_length, window):
        y = x * window
        if out_length is not None:
            in_length = x.size(-1)
            y = F.pad(y, (0, out_length - in_length))
        return y

    @staticmethod
    def _func(x, out_length, window, norm):
        window = Window._precompute(
            x.size(-1), window, norm, dtype=x.dtype, device=x.device
        )
        return Window._forward(x, out_length, window)

    @staticmethod
    def _precompute(length, window, norm, dtype=None, device=None):
        # Make window.
        params = {"dtype": dtype, "device": device}
        if window in (0, "blackman"):
            w = torch.blackman_window(length, periodic=False, **params)
        elif window in (1, "hamming"):
            w = torch.hamming_window(length, periodic=False, **params)
        elif window in (2, "hanning"):
            w = torch.hann_window(length, periodic=False, **params)
        elif window in (3, "bartlett"):
            w = torch.bartlett_window(length, periodic=False, **params)
        elif window in (4, "trapezoidal"):
            slope = torch.linspace(0, 4, length, **params)
            w = torch.minimum(torch.clip(slope, min=0, max=1), slope.flip(0))
        elif window in (5, "rectangular"):
            w = torch.ones(length, **params)
        elif window == "sine":
            w = torch.signal.windows.cosine(length, **params)
        elif window == "vorbis":
            seed = torch.signal.windows.cosine(length, **params)
            w = torch.sin(torch.pi * 0.5 * seed**2)
        elif window == "kbd":
            seed = torch.kaiser_window(length // 2 + 1, periodic=False, **params)
            cumsum = torch.cumsum(seed, dim=0)
            half = torch.sqrt(cumsum[:-1] / cumsum[-1])
            w = torch.cat([half, half.flip(0)])
        else:
            raise ValueError(f"window {window} is not supported.")

        # Normalize window.
        if norm in (0, "none"):
            pass
        elif norm in (1, "power"):
            w /= torch.sqrt(torch.sum(w**2))
        elif norm in (2, "magnitude"):
            w /= torch.sum(w)
        else:
            raise ValueError(f"norm {norm} is not supported.")

        return to(w, dtype=dtype)
