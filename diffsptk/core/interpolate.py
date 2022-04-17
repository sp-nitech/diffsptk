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
import torch.nn as nn


class Interpolation(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/interpolate.html>`_
    for details.

    Parameters
    ----------
    period : int >= 1 [scalar]
        Interpolation period, :math:`P`.

    start : int >= 0 [scalar]
        Start point, :math:`S`.

    """

    def __init__(self, period, start=0):
        super(Interpolation, self).__init__()

        self.period = period
        self.start = start

        assert 1 <= self.period
        assert 0 <= self.start

    def forward(self, x, dim=-1):
        """Interpolate signal.

        Parameters
        ----------
        x : Tensor [shape=(..., T, ...)]
            Signal.

        dim : int [scalar]
            Dimension along which to interpolate the tensors.

        Returns
        -------
        y : Tensor [shape=(..., TxP+S, ...)]
            Interpolated signal.

        Examples
        --------
        >>> x = torch.arange(1, 4)
        >>> interpolate = diffsptk.Interpolation(3, start=1)
        >>> y = interpolate(x)
        >>> y
        tensor([0, 1, 0, 0, 2, 0, 0, 3, 0, 0])

        """
        T = x.shape[dim] * self.period + self.start
        indices = torch.arange(
            self.start, T, self.period, dtype=torch.long, device=x.device
        )
        size = list(x.shape)
        size[dim] = T
        y = torch.zeros(size, dtype=x.dtype, device=x.device)
        y.index_add_(dim, indices, x)
        return y
