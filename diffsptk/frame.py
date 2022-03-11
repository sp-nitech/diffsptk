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


class Frame(nn.Module):
    def __init__(self, frame_length, frame_period, center=True):
        """Initialize module.

        Parameters
        ----------
        frame_length : int >= 1 [scalar]
            Frame length, L.

        frame_peirod : int >= 1 [scalar]
            Frame period, P.

        center : bool [scalar]
            If true, assume that the center of data is the center of frame, otherwise
            assume that the center of data is the left edge of frame.

        """
        super(Frame, self).__init__()

        self.frame_length = frame_length
        self.frame_period = frame_period
        self.center = center

        assert 1 <= self.frame_length
        assert 1 <= self.frame_period

        if self.center:
            self.left_pad_width = self.frame_length // 2
            self.right_pad_width = (self.frame_length - 1) // 2
        else:
            self.left_pad_width = 0
            self.right_pad_width = self.frame_length - 1

    def forward(self, x):
        """Apply framing to given waveform.

        Parameters
        ----------
        x : Tensor [shape=(B, T)]
            Waveform.

        Returns
        -------
        y : Tensor [shape=(B, N, L)]
            Framed waveform.

        """
        y = F.pad(x, (self.left_pad_width, self.right_pad_width))
        y = y.unfold(-1, self.frame_length, self.frame_period)
        return y
