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


class Decimation(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/decimate.html>`_
    for details.

    Parameters
    ----------
    period : int >= 1 [scalar]
        Decimation period, :math:`P`.

    start : int >= 0 [scalar]
        Start point, :math:`S`.

    """

    def __init__(self, period, start=0):
        super(Decimation, self).__init__()

        self.period = period
        self.start = start

        assert 1 <= self.period
        assert 0 <= self.start

    def forward(self, x, dim=-1):
        """Decimate signal.

        Parameters
        ----------
        x : Tensor [shape=(..., T, ...)]
            Signal.

        dim : int [scalar]
            Dimension along which to decimate the tensors.

        Returns
        -------
        y : Tensor [shape=(..., T/P-S, ...)]
            Decimated signal.

        Examples
        --------
        >>> x = torch.arange(9)
        >>> decimate = diffsptk.Decimation(3, start=1)
        >>> y = decimate(x)
        >>> y
        tensor([1, 4, 7])

        """
        T = x.shape[dim]
        indices = torch.arange(
            self.start, T, self.period, dtype=torch.long, device=x.device
        )
        y = torch.index_select(x, dim, indices)
        return y
