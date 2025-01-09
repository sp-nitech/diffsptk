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

import torch.nn.functional as F
from torch import nn

from .window import Window


class Unframe(nn.Module):
    """This is the opposite module to :func:`~diffsptk.Frame`.

    Parameters
    ----------
    frame_length : int >= 1
        Frame length, :math:`L`.

    frame_peirod : int >= 1
        Frame period, :math:`P`.

    center : bool
        If True, assume that the center of data is the center of frame, otherwise
        assume that the center of data is the left edge of frame.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular']
        Window type.

    norm : ['none', 'power', 'magnitude']
        Normalization type of window.

    """

    def __init__(
        self,
        frame_length,
        frame_period,
        *,
        center=True,
        window="rectangular",
        norm="none",
    ):
        super().__init__()

        assert 1 <= frame_period <= frame_length

        self.frame_length = frame_length
        self.frame_period = frame_period
        self.center = center
        self.register_buffer(
            "window",
            Window._precompute(self.frame_length, window, norm).view(1, -1, 1),
        )

    def forward(self, y, out_length=None):
        """Revert framed waveform.

        Parameters
        ----------
        y : Tensor [shape=(..., T/P, L)]
            Framed waveform.

        out_length : int or None
            Length of original signal, `T`.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            Waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 9)
        >>> frame = diffsptk.Frame(5, 2)
        >>> y = frame(x)
        >>> y
        tensor([[0., 0., 1., 2., 3.],
                [1., 2., 3., 4., 5.],
                [3., 4., 5., 6., 7.],
                [5., 6., 7., 8., 9.],
                [7., 8., 9., 0., 0.]])
        >>> unframe = diffsptk.Unframe(5, 2)
        >>> z = unframe(y, out_length=x.size(0))
        >>> z
        tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.])

        """
        return self._forward(
            y,
            out_length,
            self.frame_period,
            self.center,
            self.window,
        )

    @staticmethod
    def _forward(y, out_length, frame_period, center, window):
        frame_length = window.size(-2)

        d = y.dim()
        N = y.size(-2)
        assert 2 <= d <= 4, "Input must be 2D, 3D, or 4D tensor."

        def fold(x):
            x = F.fold(
                x,
                (1, (N - 1) * frame_period + frame_length),
                (1, frame_length),
                stride=(1, frame_period),
            )
            s = frame_length // 2 if center else 0
            e = None if out_length is None else s + out_length
            x = x[..., 0, 0, s:e]
            return x

        w = window.repeat(1, 1, N)
        x = y.transpose(-2, -1)

        if d == 2:
            x = x.unsqueeze(0)

        w = fold(w)
        x = fold(x)
        x = x / w

        if d == 2:
            x = x.squeeze(0)
        return x

    @staticmethod
    def _func(y, out_length, frame_length, frame_period, center, window, norm):
        window = Unframe._precompute(
            frame_length, window, norm, dtype=y.dtype, device=y.device
        )
        return Unframe._forward(y, out_length, frame_period, center, window)

    @staticmethod
    def _precompute(length, window, norm, dtype=None, device=None):
        return Window._precompute(
            length, window, norm, dtype=dtype, device=device
        ).view(1, -1, 1)
