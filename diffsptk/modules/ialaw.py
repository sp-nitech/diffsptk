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

from ..utils.private import filter_values
from .alaw import ALawCompression
from .base import BaseFunctionalModule, Precomputed


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

    def __init__(self, abs_max: float = 1, a: float = 87.6) -> None:
        super().__init__()

        self._register_precomputed(self._precompute(**filter_values(locals())))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
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
        >>> import diffsptk
        >>> alaw = diffsptk.ALawCompression(4)
        >>> ialaw = diffsptk.ALawExpansion(4)
        >>> x = diffsptk.ramp(4)
        >>> x2 = ialaw(alaw(x))
        >>> x2
        tensor([0.0000, 1.0000, 2.0000, 3.0000, 4.0000])

        """
        return self._call_forward(y)

    @staticmethod
    def _func(y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _p = ALawExpansion._precompute(*args, **kwargs)
        return ALawExpansion._apply_precomputed(_p, y=y)

    @staticmethod
    def _check(*args, **kwargs) -> None:
        ALawCompression._check(*args, **kwargs)

    @staticmethod
    def _precompute(abs_max: float, a: float) -> Precomputed:
        ALawExpansion._check(abs_max, a)
        return Precomputed(
            values={"abs_max": abs_max, "c": abs_max / a, "z": 1 + math.log(a)}
        )

    @staticmethod
    def _forward(
        y: torch.Tensor, *, abs_max: float, c: float, z: float
    ) -> torch.Tensor:
        y_abs = y.abs() / abs_max
        y1 = z * y_abs
        y2 = torch.exp(y1 - 1)
        condition = y_abs < 1 / z
        x = c * torch.sign(y) * torch.where(condition, y1, y2)
        return x
