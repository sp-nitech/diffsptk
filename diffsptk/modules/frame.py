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


class Frame(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/frame.html>`_
    for details.

    Parameters
    ----------
    frame_length : int >= 1 [scalar]
        Frame length, :math:`L`.

    frame_peirod : int >= 1 [scalar]
        Frame period, :math:`P`.

    center : bool [scalar]
        If True, assume that the center of data is the center of frame, otherwise
        assume that the center of data is the left edge of frame.

    zmean : bool [scalar]
        If True, perform mean subtraction on each frame.

    """

    def __init__(self, frame_length, frame_period, center=True, zmean=False):
        super(Frame, self).__init__()

        self.frame_length = frame_length
        self.frame_period = frame_period
        self.zmean = zmean

        assert 1 <= self.frame_length
        assert 1 <= self.frame_period

        # Make padding module.
        if center:
            left_pad_width = self.frame_length // 2
            right_pad_width = (self.frame_length - 1) // 2
        else:
            left_pad_width = 0
            right_pad_width = self.frame_length - 1

        self.pad = nn.ConstantPad1d((left_pad_width, right_pad_width), 0)

    def forward(self, x):
        """Apply framing to given waveform.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Waveform.

        Returns
        -------
        y : Tensor [shape=(..., T/P, L)]
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
        y = self.pad(x)
        y = y.unfold(-1, self.frame_length, self.frame_period)
        if self.zmean:
            y = y - y.mean(-1, keepdim=True)
        return y
