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

from ..typing import Precomputed
from ..utils.private import filter_values
from .base import BaseFunctionalModule
from .decimate import Decimation


class Interpolation(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/interpolate.html>`_
    for details.

    Parameters
    ----------
    period : int >= 1
        The interpolation period, :math:`P`.

    start : int >= 0
        The start point, :math:`S`.

    dim : int
        The dimension along which to interpolate the tensors.

    """

    def __init__(self, period: int, start: int = 0, dim: int = -1) -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate the input signal.

        Parameters
        ----------
        x : Tensor [shape=(..., T, ...)]
            The input signal.

        Returns
        -------
        out : Tensor [shape=(..., TxP+S, ...)]
            The interpolated signal.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 3)
        >>> interpolate = diffsptk.Interpolation(3, start=1)
        >>> y = interpolate(x)
        >>> y
        tensor([0., 1., 0., 0., 2., 0., 0., 3., 0., 0.])

        """
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = Interpolation._precompute(*args, **kwargs)
        return Interpolation._forward(x, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(*args, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def _precompute(*args, **kwargs) -> Precomputed:
        return Decimation._precompute(*args, **kwargs)

    @staticmethod
    def _forward(x: torch.Tensor, period: int, start: int, dim: int) -> torch.Tensor:
        if not -x.ndim <= dim < x.ndim:
            raise ValueError(f"Dimension {dim} out of range.")

        T = x.shape[dim] * period + start
        output_size = list(x.shape)
        output_size[dim] = T

        y = torch.zeros(output_size, device=x.device, dtype=x.dtype)
        indices = torch.arange(start, T, period, device=x.device, dtype=torch.long)
        y.index_copy_(dim, indices, x)
        return y
