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


class Decimation(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/decimate.html>`_
    for details.

    Parameters
    ----------
    period : int >= 1
        Decimation period, :math:`P`.

    start : int >= 0
        Start point, :math:`S`.

    dim : int
        Dimension along which to shift the tensors.

    """

    def __init__(self, period, start=0, dim=-1):
        super().__init__()

        assert 1 <= period
        assert 0 <= start

        self.period = period
        self.start = start
        self.dim = dim

    def forward(self, x):
        """Decimate signal.

        Parameters
        ----------
        x : Tensor [shape=(..., T, ...)]
            Signal.

        dim : int
            Dimension along which to decimate the tensors.

        Returns
        -------
        out : Tensor [shape=(..., T/P-S, ...)]
            Decimated signal.

        Examples
        --------
        >>> x = diffsptk.ramp(9)
        >>> decimate = diffsptk.Decimation(3, start=1)
        >>> y = decimate(x)
        >>> y
        tensor([1., 4., 7.])

        """
        return self._forward(x, self.period, self.start, self.dim)

    @staticmethod
    def _forward(x, period, start, dim):
        T = x.shape[dim]
        indices = torch.arange(start, T, period, dtype=torch.long, device=x.device)
        y = torch.index_select(x, dim, indices)
        return y

    _func = _forward
