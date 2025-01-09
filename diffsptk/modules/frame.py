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


class Frame(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/frame.html>`_
    for details.

    Parameters
    ----------
    frame_length : int >= 1
        Frame length, :math:`L`.

    frame_peirod : int >= 1
        Frame period, :math:`P`.

    center : bool
        If True, assume that the center of data is the center of frame, otherwise
        assume that the center of data is the left edge of frame.

    zmean : bool
        If True, perform mean subtraction on each frame.

    """

    def __init__(self, frame_length, frame_period, center=True, zmean=False):
        super().__init__()

        assert 1 <= frame_length
        assert 1 <= frame_period

        self.frame_length = frame_length
        self.frame_period = frame_period
        self.center = center
        self.zmean = zmean

    def forward(self, x):
        """Apply framing to given waveform.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Waveform.

        Returns
        -------
        out : Tensor [shape=(..., T/P, L)]
            Framed waveform.

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

        """
        return self._forward(
            x, self.frame_length, self.frame_period, self.center, self.zmean
        )

    @staticmethod
    def _forward(x, frame_length, frame_period, center, zmean):
        if center:
            padding = (frame_length // 2, (frame_length - 1) // 2)
        else:
            padding = (0, frame_length - 1)
        x = F.pad(x, padding)
        y = x.unfold(-1, frame_length, frame_period)
        if zmean:
            y = y - y.mean(-1, keepdim=True)
        return y

    _func = _forward
