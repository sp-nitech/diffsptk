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
from torch import nn


class ALawCompression(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/alaw.html>`_
    for details.

    Parameters
    ----------
    abs_max : float > 0
        Absolute maximum value of input.

    a : float >= 1
        Compression factor, :math:`A`.

    """

    def __init__(self, abs_max=1, a=87.6):
        super().__init__()

        assert 0 < abs_max
        assert 1 <= a

        self.abs_max = abs_max
        self.a = a
        self.const = self._precompute(self.abs_max, self.a)

    def forward(self, x):
        """Compress waveform by A-law algorithm.

        Parameters
        ----------
        x : Tensor [shape=(...,)]
            Waveform.

        Returns
        -------
        out : Tensor [shape=(...,)]
            Compressed waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> alaw = diffsptk.ALawCompression(4)
        >>> y = alaw(x)
        >>> y
        tensor([0.0000, 2.9868, 3.4934, 3.7897, 4.0000])

        """
        return self._forward(x, self.abs_max, self.a, self.const)

    @staticmethod
    def _forward(x, abs_max, a, const):
        x_abs = x.abs() / abs_max
        x1 = a * x_abs
        x2 = 1 + torch.log(x1)
        condition = x_abs < 1 / a
        y = const * torch.sign(x) * torch.where(condition, x1, x2)
        return y

    @staticmethod
    def _func(x, abs_max, a):
        const = ALawCompression._precompute(abs_max, a)
        return ALawCompression._forward(x, abs_max, a, const)

    @staticmethod
    def _precompute(abs_max, a):
        return abs_max / (1 + math.log(a))
