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
import torch.nn as nn

from ..misc.utils import replicate1


class ZeroCrossingAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/zcross.html>`_
    for details. **Note that this module cannot compute gradient**.

    Parameters
    ----------
    frame_length : int >= 1 [scalar]
        Frame length, :math:`L`.

    norm : bool [scalar]
        If True, divide zero-crossing rate by frame length.

    """

    def __init__(self, frame_length, norm=False):
        super(ZeroCrossingAnalysis, self).__init__()

        self.frame_length = frame_length
        self.norm = norm

        assert 1 <= self.frame_length

    def forward(self, x):
        """Compute zero-crossing rate.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Waveform.

        Returns
        -------
        z : Tensor [shape=(..., T/L)]
            Zero-crossing rate.

        Examples
        --------
        >>> x = diffsptk.nrand(5)
        >>> x
        tensor([-0.2388,  0.3587, -0.6606, -0.6929,  0.5239,  0.4501])
        >>> zcross = diffsptk.ZeroCrossingAnalysis(3)
        >>> z = zcross(x)
        >>> z
        tensor([2., 1.])

        """
        x = torch.sign(x)
        x = replicate1(x, right=False)
        x = x.unfold(-1, self.frame_length + 1, self.frame_length)
        z = 0.5 * (x[..., 1:] - x[..., :-1]).abs().sum(-1)
        if self.norm:
            z = z / self.frame_length
        return z
