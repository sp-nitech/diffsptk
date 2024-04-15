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
from torch import nn


class Interpolation(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/interpolate.html>`_
    for details.

    Parameters
    ----------
    period : int >= 1
        Interpolation period, :math:`P`.

    start : int >= 0
        Start point, :math:`S`.

    dim : int
        Dimension along which to interpolate the tensors.

    """

    def __init__(self, period, start=0, dim=-1):
        super().__init__()

        assert 1 <= period
        assert 0 <= start

        self.period = period
        self.start = start
        self.dim = dim

    def forward(self, x):
        """Interpolate signal.

        Parameters
        ----------
        x : Tensor [shape=(..., T, ...)]
            Signal.

        Returns
        -------
        out : Tensor [shape=(..., TxP+S, ...)]
            Interpolated signal.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 3)
        >>> interpolate = diffsptk.Interpolation(3, start=1)
        >>> y = interpolate(x)
        >>> y
        tensor([0., 1., 0., 0., 2., 0., 0., 3., 0., 0.])

        """
        return self._forward(x, self.period, self.start, self.dim)

    @staticmethod
    def _forward(x, period, start, dim):
        # Determine the size of the output tensor.
        T = x.shape[dim] * period + start
        size = list(x.shape)
        size[dim] = T

        y = torch.zeros(size, dtype=x.dtype, device=x.device)
        indices = torch.arange(start, T, period, dtype=torch.long, device=x.device)
        y.index_add_(dim, indices, x)
        return y

    _func = _forward
