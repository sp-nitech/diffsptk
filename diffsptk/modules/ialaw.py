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

from .alaw import ALawCompression
from .base import BaseFunctionalModule


class ALawExpansion(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ialaw.html>`_
    for details.

    Parameters
    ----------
    abs_max : float > 0
        The absolute maximum value of the original input waveform.

    a : float >= 1
        The compression factor, :math:`A`.

    """

    def __init__(self, abs_max=1, a=87.6):
        super().__init__()

        self.values = self._precompute(abs_max, a)

    def forward(self, y):
        """Expand the waveform using the A-law algorithm.

        Parameters
        ----------
        y : Tensor [shape=(...,)]
            The input compressed waveform.

        Returns
        -------
        out : Tensor [shape=(...,)]
            The expanded waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> alaw = diffsptk.ALawCompression(4)
        >>> ialaw = diffsptk.ALawExpansion(4)
        >>> x2 = ialaw(alaw(x))
        >>> x2
        tensor([0.0000, 1.0000, 2.0000, 3.0000, 4.0000])

        """
        return self._forward(y, *self.values)

    @staticmethod
    def _func(y, *args, **kwargs):
        values = ALawExpansion._precompute(*args, **kwargs)
        return ALawExpansion._forward(y, *values)

    @staticmethod
    def _check(abs_max, a):
        ALawCompression._check(abs_max, a)

    @staticmethod
    def _precompute(abs_max, a):
        ALawExpansion._check(abs_max, a)
        return (
            abs_max,
            abs_max / a,
            1 + math.log(a),
        )

    @staticmethod
    def _forward(y, abs_max, c, z):
        y_abs = y.abs() / abs_max
        y1 = z * y_abs
        y2 = torch.exp(y1 - 1)
        condition = y_abs < 1 / z
        x = c * torch.sign(y) * torch.where(condition, y1, y2)
        return x
