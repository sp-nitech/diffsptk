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

import torch.nn as nn
import torch.nn.functional as F

from .window import Window


class Unframe(nn.Module):
    """This is the opposite module to :func:`~diffsptk.Frame`.

    Parameters
    ----------
    frame_length : int >= 1 [scalar]
        Frame length, :math:`L`.

    frame_peirod : int >= 1 [scalar]
        Frame period, :math:`P`.

    center : bool [scalar]
        If True, assume that the center of data is the center of frame, otherwise
        assume that the center of data is the left edge of frame.

    norm : ['none', 'power', 'magnitude']
        Normalization type of window.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular']
        Window type.

    """

    def __init__(
        self,
        frame_length,
        frame_period,
        center=True,
        norm="none",
        window="rectangular",
    ):
        super(Unframe, self).__init__()

        self.frame_length = frame_length
        self.frame_period = frame_period

        assert 1 <= self.frame_length
        assert 1 <= self.frame_period

        if center:
            self.left_pad_width = self.frame_length // 2
        else:
            self.left_pad_width = 0

        self.register_buffer(
            "window",
            Window(frame_length, window=window, norm=norm).window.view(1, -1, 1),
        )

    def forward(self, y, out_length=None):
        """Revert framed waveform.

        Parameters
        ----------
        y : Tensor [shape=(..., T/P, L)]
            Framed waveform.

        out_length : int [scalar]
            Length of original signal, `T`.

        Returns
        -------
        x : Tensor [shape=(..., T)]
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
        d = y.dim()
        N = y.size(-2)
        assert 2 <= d <= 4, "Input must be 2D, 3D, or 4D tensor"

        def fold(x):
            x = F.fold(
                x,
                (1, (N - 1) * self.frame_period + self.frame_length),
                (1, self.frame_length),
                stride=(1, self.frame_period),
            )
            s = self.left_pad_width
            e = None if out_length is None else s + out_length
            x = x[..., 0, 0, s:e]
            return x

        w = self.window.repeat(1, 1, N)
        x = y.mT

        if d == 2:
            x = x.unsqueeze(0)

        w = fold(w)
        x = fold(x)
        x = x / w

        if d == 2:
            x = x.squeeze(0)
        return x
