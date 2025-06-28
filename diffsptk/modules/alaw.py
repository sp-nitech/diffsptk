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

from ..typing import Precomputed
from ..utils.private import filter_values
from .base import BaseFunctionalModule


class ALawCompression(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/alaw.html>`_
    for details.

    Parameters
    ----------
    abs_max : float > 0
        The absolute maximum value of the input waveform.

    a : float >= 1
        The compression factor, :math:`A`.

    """

    def __init__(self, abs_max: float = 1, a: float = 87.6) -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress the input waveform using the A-law algorithm.

        Parameters
        ----------
        x : Tensor [shape=(...,)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(...,)]
            The compressed waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> alaw = diffsptk.ALawCompression(4)
        >>> y = alaw(x)
        >>> y
        tensor([0.0000, 2.9868, 3.4934, 3.7897, 4.0000])

        """
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = ALawCompression._precompute(*args, **kwargs)
        return ALawCompression._forward(x, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(abs_max: float, a: float) -> None:
        if abs_max < 0:
            raise ValueError("abs_max must be non-negative.")
        if a < 1:
            raise ValueError("a must be greater than or equal to 1.")

    @staticmethod
    def _precompute(abs_max: float, a: float) -> Precomputed:
        ALawCompression._check(abs_max, a)
        return (
            abs_max,
            a,
            abs_max / (1 + math.log(a)),
        )

    @staticmethod
    def _forward(x: torch.Tensor, abs_max: float, a: float, c: float) -> torch.Tensor:
        x_abs = x.abs() / abs_max
        x1 = a * x_abs
        x2 = 1 + torch.log(x1)
        condition = x_abs < 1 / a
        y = c * torch.sign(x) * torch.where(condition, x1, x2)
        return y
