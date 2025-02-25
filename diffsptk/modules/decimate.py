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

from ..misc.utils import get_values
from .base import BaseFunctionalModule


class Decimation(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/decimate.html>`_
    for details.

    Parameters
    ----------
    period : int >= 1
        The decimation period, :math:`P`.

    start : int >= 0
        The start point, :math:`S`.

    dim : int
        The dimension along which to shift the tensors.

    """

    def __init__(self, period, start=0, dim=-1):
        super().__init__()

        self.values = self._precompute(*get_values(locals()))

    def forward(self, x):
        """Decimate the input signal.

        Parameters
        ----------
        x : Tensor [shape=(..., T, ...)]
            The input signal.

        Returns
        -------
        out : Tensor [shape=(..., T/P-S, ...)]
            The decimated signal.

        Examples
        --------
        >>> x = diffsptk.ramp(9)
        >>> decimate = diffsptk.Decimation(3, start=1)
        >>> y = decimate(x)
        >>> y
        tensor([1., 4., 7.])

        """
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x, *args, **kwargs):
        values = Decimation._precompute(*args, **kwargs)
        return Decimation._forward(x, *values)

    @staticmethod
    def _check(period, start, dim):
        if period <= 0:
            raise ValueError("period must be positive.")
        if start < 0:
            raise ValueError("start must be non-negative.")

    @staticmethod
    def _precompute(period, start, dim):
        Decimation._check(period, start, dim)
        return period, start, dim

    @staticmethod
    def _forward(x, period, start, dim):
        if not -x.ndim <= dim < x.ndim:
            raise ValueError(f"Dimension {dim} out of range.")
        dim = dim % x.ndim  # Handle negative dim.
        y = x[(slice(None),) * dim + (slice(start, None, period),)]
        return y
