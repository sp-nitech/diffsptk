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
import torch.nn.functional as F

from ..typing import Precomputed
from ..utils.private import filter_values
from .base import BaseFunctionalModule


class Delay(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/delay.html>`_
    for details.

    Parameters
    ----------
    start : int
        The start point, :math:`S`. If negative, advance the signal.

    keeplen : bool
        If True, the output has the same length of the input.

    dim : int
        The dimension along which to delay the tensors.

    """

    def __init__(self, start: int, keeplen: bool = False, dim: int = -1) -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delay the input signal.

        Parameters
        ----------
        x : Tensor [shape=(..., T, ...)]
            The input signal.

        Returns
        -------
        out : Tensor [shape=(..., T-S, ...)] or [shape=(..., T, ...)]
            The delayed signal.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 3)
        >>> delay = diffsptk.Delay(2)
        >>> y = delay(x)
        >>> y
        tensor([0., 0., 1., 2., 3.])

        """
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = Delay._precompute(*args, **kwargs)
        return Delay._forward(x, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check() -> None:
        pass

    @staticmethod
    def _precompute(start: int, keeplen: bool, dim: int) -> Precomputed:
        Delay._check()
        return start, keeplen, dim

    @staticmethod
    def _forward(x: torch.Tensor, start: int, keeplen: bool, dim: int) -> torch.Tensor:
        if not -x.ndim <= dim < x.ndim:
            raise ValueError(f"Dimension {dim} out of range.")

        if start == 0:
            return x

        dim = dim % x.ndim
        pad = [0] * (2 * x.ndim)
        if 0 < start:
            # Delay case:
            pad[2 * (x.ndim - 1 - dim)] = start
            y = F.pad(x, pad)
            if keeplen:
                y = y.narrow(dim, 0, x.size(dim))
        else:
            # Advance case:
            y = x.narrow(dim, -start, x.size(dim) + start)
            if keeplen:
                pad[2 * (x.ndim - 1 - dim) + 1] = -start
                y = F.pad(y, pad)
        return y
