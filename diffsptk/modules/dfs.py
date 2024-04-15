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
import torch.nn.functional as F

from ..misc.utils import iir
from ..misc.utils import to
from ..misc.utils import to_3d


class InfiniteImpulseResponseDigitalFilter(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/dfs.html>`_
    for details.

    Parameters
    ----------
    b : List [shape=(M+1,)] or None
        Numerator coefficients.

    a : List [shape=(N+1,)] or None
        Denominator coefficients.

    ir_length : int >= 1
        Length of impulse response.

    learnable : bool
        If True, the filter coefficients are learnable.

    """

    def __init__(self, b=None, a=None, ir_length=None, learnable=False):
        super().__init__()

        if b is None:
            b = [1]
        if a is None:
            a = [1]
        b = torch.tensor(b)
        a = torch.tensor(a)

        if ir_length is None:
            ir_length = len(b)
        assert 1 <= ir_length

        # Pre-compute impulse response.
        d = torch.zeros(max(len(b), len(a)), dtype=torch.double)
        h = torch.empty(ir_length, dtype=torch.double)
        a0 = a[0]
        a1 = a[1:]
        for t in range(ir_length):
            x = a0 if t == 0 else 0
            y = x - torch.sum(d[: len(a1)] * a1)
            d = torch.roll(d, 1)
            d[0] = y
            y = torch.sum(d[: len(b)] * b)
            h[t] = y
        h = to(h.reshape(1, 1, -1).flip(-1))
        if learnable:
            self.h = nn.Parameter(h)
        else:
            self.register_buffer("h", h)

    def forward(self, x):
        """Apply an IIR digital filter.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Input waveform.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            Filtered waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> dfs = diffsptk.IIR(b=[1, -0.97])
        >>> y = dfs(x)
        >>> y
        tensor([0.0000, 1.0000, 1.0300, 1.0600, 1.0900])

        """
        return self._forward(x, self.h)

    @staticmethod
    def _forward(x, h):
        y = to_3d(x)
        y = F.pad(y, (h.size(-1) - 1, 0))
        y = F.conv1d(y, h)
        y = y.view_as(x)
        return y

    @staticmethod
    def _func(x, b=None, a=None):
        return iir(x, b, a)
