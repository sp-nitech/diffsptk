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

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..misc.utils import numpy_to_torch


class InfiniteImpulseResponseDigitalFilter(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/dfs.html>`_
    for details. Note that numerator and denominator coefficients are fixed in the
    current implementation.

    Parameters
    ----------
    b : List [shape=(M+1,)]
        Numerator coefficients.

    a : List [shape=(N+1,)]
        Denominator coefficients.

    ir_length : int >= 1 [scalar]
        Length of impulse response.

    """

    def __init__(self, b=[1], a=[1], ir_length=None):
        super(InfiniteImpulseResponseDigitalFilter, self).__init__()

        if ir_length is None:
            ir_length = len(b)
        assert 1 <= ir_length

        d = np.zeros(max(len(b), len(a)))
        h = np.empty(ir_length)

        a0 = a[0]
        a1 = np.asarray(a[1:])
        b = np.asarray(b)

        # Pre-compute impulse response.
        for t in range(ir_length):
            x = a0 if t == 0 else 0
            y = x - np.sum(d[: len(a1)] * a1)

            d = np.roll(d, 1)
            d[0] = y

            y = np.sum(d[: len(b)] * b)
            h[t] = y

        h = h.reshape(1, 1, -1)
        self.register_buffer("h", numpy_to_torch(h).flip(-1))

        self.pad = nn.ConstantPad1d((ir_length - 1, 0), 0)

    def forward(self, x):
        """Apply an approximated IIR digital filter.

        Parameters
        ----------
        x : Tensor [shape=(B, 1, T) or (B, T) or (T,)]
            Input waveform.

        Returns
        -------
        y : Tensor [shape=(B, 1, T) or (B, T) or (T,)]
            Filterd waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> dfs = diffsptk.IIR(b=[1, -0.97])
        >>> y = dfs(x)
        >>> y
        tensor([0.0000, 1.0000, 1.0300, 1.0600, 1.0900])

        """
        d = x.dim()
        if d == 1:
            x = x.view(1, 1, -1)
        elif d == 2:
            x = x.unsqueeze(1)
        assert x.dim() == 3, "Input must be 3D tensor"

        y = F.conv1d(self.pad(x), self.h)

        if d == 1:
            y = y.view(-1)
        elif d == 2:
            y = y.squeeze(1)
        return y
