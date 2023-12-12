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

from ..misc.utils import iir
from ..misc.utils import numpy_to_torch
from ..misc.utils import to_3d


class InfiniteImpulseResponseDigitalFilter(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/dfs.html>`_
    for details.

    Parameters
    ----------
    b : List [shape=(M+1,)]
        Numerator coefficients.

    a : List [shape=(N+1,)]
        Denominator coefficients.

    ir_length : int >= 1 [scalar]
        Length of impulse response (valid only if **mode** is 'fir').

    mode : ['fir', 'iir']
        If 'fir', filter is approximated by a finite impulse response.

    """

    def __init__(self, b=None, a=None, ir_length=None, mode="fir"):
        super(InfiniteImpulseResponseDigitalFilter, self).__init__()

        self.mode = mode

        if b is None:
            b = [1]
        if a is None:
            a = [1]
        b = np.asarray(b)
        a = np.asarray(a)

        if self.mode == "fir":
            # Pre-compute impulse response.
            if ir_length is None:
                ir_length = len(b)
            assert 1 <= ir_length

            d = np.zeros(max(len(b), len(a)))
            h = np.empty(ir_length)
            a0 = a[0]
            a1 = a[1:]
            for t in range(ir_length):
                x = a0 if t == 0 else 0
                y = x - np.sum(d[: len(a1)] * a1)

                d = np.roll(d, 1)
                d[0] = y

                y = np.sum(d[: len(b)] * b)
                h[t] = y
            h = h.reshape(1, 1, -1)
            self.register_buffer("h", numpy_to_torch(h).flip(-1))
        elif self.mode == "iir":
            self.register_buffer("b", numpy_to_torch(b))
            self.register_buffer("a", numpy_to_torch(a))
        else:
            raise ValueError(f"mode {mode} is not supported")

    def forward(self, x, b=None, a=None):
        """Apply an IIR digital filter.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Input waveform.

        b : Tensor [shape=(M+1,)]
            Numerator coefficients.

        a : Tensor [shape=(N+1,)]
            Denominator coefficients.

        Returns
        -------
        y : Tensor [shape=(..., T)]
            Filtered waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> dfs = diffsptk.IIR(b=[1, -0.97])
        >>> y = dfs(x)
        >>> y
        tensor([0.0000, 1.0000, 1.0300, 1.0600, 1.0900])

        """
        if self.mode == "fir":
            y = self._forward_fir(x, b, a)
        elif self.mode == "iir":
            y = self._forward_iir(x, b, a)
        else:
            raise RuntimeError
        return y

    def _forward_fir(self, x, b=None, a=None):
        if a is None and b is None:
            h = self.h
        elif a is None and b is not None:
            h = b.view(1, 1, -1).flip(-1)
        else:
            raise ValueError("Denominator coefficients must be set via constructor")

        y = to_3d(x)
        y = F.pad(y, (h.size(-1) - 1, 0))
        y = F.conv1d(y, h)
        y = y.view_as(x)
        return y

    def _forward_iir(self, x, b=None, a=None):
        if b is None:
            b = self.b
        if a is None:
            a = self.a

        y = iir(x, b, a)
        return y
