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


class ALawExpansion(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ialaw.html>`_
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

    def forward(self, y):
        """Expand waveform by A-law algorithm.

        Parameters
        ----------
        y : Tensor [shape=(...,)]
            Compressed waveform.

        Returns
        -------
        out : Tensor [shape=(...,)]
            Waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> alaw = diffsptk.ALawCompression(4)
        >>> ialaw = diffsptk.ALawExpansion(4)
        >>> x2 = ialaw(alaw(x))
        >>> x2
        tensor([0.0000, 1.0000, 2.0000, 3.0000, 4.0000])

        """
        return self._forward(y, self.abs_max, *self.const)

    @staticmethod
    def _forward(y, abs_max, const, z):
        y_abs = y.abs() / abs_max
        y1 = z * y_abs
        y2 = torch.exp(y1 - 1)
        condition = y_abs < 1 / z
        x = const * torch.sign(y) * torch.where(condition, y1, y2)
        return x

    @staticmethod
    def _func(y, abs_max, a):
        const = ALawExpansion._precompute(abs_max, a)
        return ALawExpansion._forward(y, abs_max, *const)

    @staticmethod
    def _precompute(abs_max, a):
        return abs_max / a, 1 + math.log(a)
