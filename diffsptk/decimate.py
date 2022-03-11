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
    def __init__(self, period, start=0):
        """Initialize module.

        Parameters
        ----------
        peirod : int >= 1 [scalar]
            Decimation period, P.

        start : int >= 0 [scalar]
            Start point, S.

        """
        super(Decimation, self).__init__()

        self.period = period
        self.start = start

        assert 1 <= self.period
        assert 0 <= self.start

    def forward(self, x, dim=0):
        """Decimate signal.

        Parameters
        ----------
        x : Tensor [shape=(..., T, ...)]
            Signal.

        dim : int [scalar]
            Dimension along which to decimate the elements.

        Returns
        -------
        y : Tensor [shape=(..., T/P-S, ...)]
            Decimated signal.

        """
        indices = torch.arange(
            self.start, x.shape[dim], self.period, dtype=torch.long, device=x.device
        )
        y = torch.index_select(x, dim, indices)
        return y
