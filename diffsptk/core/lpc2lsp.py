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
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..misc.utils import TWO_PI
from ..misc.utils import check_size
from ..misc.utils import numpy_to_torch


class LinearPredictiveCoefficientsToLineSpectralPairs(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lpc2lsp.html>`_
    for details. **Note that this module cannot compute gradient**.

    Parameters
    ----------
    lpc_order : int >= 0 [scalar]
        Order of LPC, :math:`M`.

    n_split : int >= 1 [scalar]
        Number of splits of unit semicircle.

    n_iter : int >= 0 [scalar]
        Number of pseudo iterations.

    log_gain : bool [scalar]
        If True, output gain in log scale.

    sample_rate : int >= 1 [scalar]
        Sample rate in Hz.

    out_format : ['radian', 'cycle', 'khz', 'hz']
        Output format.

    """

    def __init__(
        self,
        lpc_order,
        n_split=512,
        n_iter=0,
        log_gain=False,
        sample_rate=None,
        out_format="radian",
    ):
        super(LinearPredictiveCoefficientsToLineSpectralPairs, self).__init__()

        self.lpc_order = lpc_order
        self.log_gain = log_gain

        assert 0 <= self.lpc_order < n_split
        assert 0 <= n_iter

        if self.lpc_order % 2 == 0:
            sign = np.ones(self.lpc_order // 2 + 2)
            sign[::2] = -1
            self.register_buffer("sign", numpy_to_torch(sign))
            mask = np.ones(self.lpc_order // 2 + 2)
            mask[::2] = 0
            self.register_buffer("mask", numpy_to_torch(mask))

        x = np.linspace(1, -1, n_split * (n_iter + 1) + 1)
        self.register_buffer("x", numpy_to_torch(x))

        # Avoid the use of Chebyshev polynomials.
        omega = np.arccos(x)
        k = np.arange(self.lpc_order // 2 + 2)
        Tx = np.cos(k.reshape(-1, 1) * omega.reshape(1, -1))
        scale = np.ones(self.lpc_order // 2 + 2)
        scale[0] = 0.5
        Tx = scale.reshape(-1, 1) * Tx
        self.register_buffer("Tx", numpy_to_torch(Tx))

        if out_format == 0 or out_format == "radian":
            self.convert = lambda x: x
        elif out_format == 1 or out_format == "cycle":
            self.convert = lambda x: x / TWO_PI
        elif out_format == 2 or out_format == "khz":
            assert sample_rate is not None and 0 < sample_rate
            self.convert = lambda x: x * (sample_rate / 1000 / TWO_PI)
        elif out_format == 3 or out_format == "hz":
            assert sample_rate is not None and 0 < sample_rate
            self.convert = lambda x: x * (sample_rate / TWO_PI)
        else:
            raise ValueError(f"out_format {out_format} is not supported")

    def forward(self, a):
        """Convert LPC to LSP.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            LPC coefficients.

        Returns
        -------
        w : Tensor [shape=(..., M+1)]
            LSP coefficients.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([-1.5326,  1.0875, -1.5925,  0.6913,  1.6217])
        >>> lpc = diffsptk.LPC(3, 5)
        >>> a = lpc(x)
        >>> a
        tensor([ 2.7969,  0.3908,  0.0458, -0.0859])
        >>> lpc2lsp = diffsptk.LinearPredictiveCoefficientsToLineSpectralPairs(3)
        >>> w = lpc2lsp(a)
        >>> w
        tensor([2.7969, 0.9037, 1.8114, 2.4514])

        """
        check_size(a.size(-1), self.lpc_order + 1, "dimension of LPC")

        K, a = torch.split(a, [1, self.lpc_order], dim=-1)

        p1 = a[..., : (self.lpc_order + 1) // 2]
        p2 = a.flip(-1)[..., : (self.lpc_order + 1) // 2]
        q1 = p1 + p2
        q2 = p1 - p2
        if self.lpc_order % 2 == 0:
            d1 = F.pad(q1, (1, 0), value=1)
            d2 = F.pad(q2, (1, 0), value=1)
            c1_odd = torch.cumsum(d1 * self.sign[:-1], dim=-1)
            c1_even = torch.cumsum(d1 * self.sign[1:], dim=-1)
            c1 = c1_odd * self.mask[:-1] + c1_even * self.mask[1:]
            c2 = torch.cumsum(d2, dim=-1)
        elif self.lpc_order == 1:
            c1 = F.pad(q1, (1, 0), value=1)
            c2 = c1
        else:
            d1 = F.pad(q1, (1, 0), value=1)
            d2_odd = F.pad(q2[..., 0::2], (1, 0), value=0)
            d2_even = F.pad(q2[..., 1::2], (1, 0), value=1)
            c1 = d1
            c2_odd = torch.cumsum(d2_odd, dim=-1)
            c2_even = torch.cumsum(d2_even, dim=-1)
            c2 = torch.flatten(torch.stack([c2_odd, c2_even], dim=-1), start_dim=-2)
            c2 = c2[..., 1:-1]
        c1 = c1.flip(-1)
        c2 = c2.flip(-1)

        y1 = torch.matmul(c1, self.Tx[: c1.size(-1)])
        y2 = torch.matmul(c2, self.Tx[: c2.size(-1)])

        index1 = y1[..., :-1] * y1[..., 1:] <= 0
        index2 = y2[..., :-1] * y2[..., 1:] <= 0
        index = torch.logical_or(index1, index2)

        i1 = F.pad(index1, (0, 1), value=False)
        i2 = F.pad(index2, (0, 1), value=False)
        i1 = torch.logical_or(i1, torch.roll(i1, 1, dims=-1))
        i2 = torch.logical_or(i2, torch.roll(i2, 1, dims=-1))
        y = y1 * i1 + y2 * i2

        x_upper = torch.masked_select(self.x[:-1], index)
        x_lower = torch.masked_select(self.x[1:], index)
        y_upper = torch.masked_select(y[..., :-1], index)
        y_lower = torch.masked_select(y[..., 1:], index)
        x = (y_lower * x_upper - y_upper * x_lower) / (y_lower - y_upper)
        w = torch.acos(x).view_as(a)

        w = self.convert(w)
        if self.log_gain:
            K = torch.log(K)
        w = torch.cat([K, w], dim=-1)
        return w
