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

import math

import torch
import torch.nn as nn


class ALawCompression(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/alaw.html>`_
    for details.

    Parameters
    ----------
    abs_max : float > 0 [scalar]
        Absolute maximum value of input.

    a : float >= 1 [scalar]
        Compression factor, :math:`A`.

    """

    def __init__(self, abs_max=1, a=87.6):
        super(ALawCompression, self).__init__()

        self.abs_max = abs_max
        self.a = a

        assert 0 < self.abs_max
        assert 1 <= self.a

        self.const = self.abs_max / (1 + math.log(self.a))

    def forward(self, x):
        """Compress waveform by A-law algorithm.

        Parameters
        ----------
        x : Tensor [shape=(...,)]
            Waveform.

        Returns
        -------
        y : Tensor [shape=(...,)]
            Compressed waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> alaw = diffsptk.ALawCompression(4)
        >>> y = alaw(x)
        >>> y
        tensor([0.0000, 2.9868, 3.4934, 3.7897, 4.0000])

        """
        x_abs = x.abs() / self.abs_max
        x1 = self.a * x_abs
        x2 = 1 + torch.log(x1)
        condition = x_abs < (1 / self.a)
        y = self.const * torch.sign(x) * torch.where(condition, x1, x2)
        return y
