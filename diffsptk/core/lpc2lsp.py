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
import torch.nn.functional as F

from ..misc.utils import TWO_PI
from ..misc.utils import check_size
from ..misc.utils import deconv1d
from ..misc.utils import numpy_to_torch
from .root_pol import PolynomialToRoots


class LinearPredictiveCoefficientsToLineSpectralPairs(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lpc2lsp.html>`_
    for details.

    Parameters
    ----------
    lpc_order : int >= 0 [scalar]
        Order of LPC, :math:`M`.

    log_gain : bool [scalar]
        If True, output gain in log scale.

    sample_rate : int >= 1 [scalar]
        Sample rate in Hz.

    out_format : ['radian', 'cycle', 'khz', 'hz']
        Output format.

    """

    def __init__(
        self, lpc_order, log_gain=False, sample_rate=None, out_format="radian"
    ):
        super(LinearPredictiveCoefficientsToLineSpectralPairs, self).__init__()

        self.lpc_order = lpc_order
        self.log_gain = log_gain

        assert 0 <= self.lpc_order

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

        if self.lpc_order == 0:
            pass
        elif self.lpc_order == 1:
            self.root_q = PolynomialToRoots(lpc_order + 1)
        elif self.lpc_order % 2 == 0:
            self.root_p = PolynomialToRoots(lpc_order)
            self.root_q = PolynomialToRoots(lpc_order)
            self.register_buffer("kernel_p", numpy_to_torch([1, -1]))
            self.register_buffer("kernel_q", numpy_to_torch([1, 1]))
        else:
            self.root_p = PolynomialToRoots(lpc_order - 1)
            self.root_q = PolynomialToRoots(lpc_order + 1)
            self.register_buffer("kernel_p", numpy_to_torch([1, 0, -1]))
            self.register_buffer("kernel_q", numpy_to_torch([1]))

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

        if self.log_gain:
            K = torch.log(K)
        if self.lpc_order == 0:
            return K

        a0 = F.pad(a, (1, 0), value=1)
        a1 = F.pad(a0, (0, 1), value=0)
        a2 = a1.flip(-1)
        p = a1 - a2
        q = a1 + a2
        if self.lpc_order == 1:
            q = self.root_q(q)
            w = torch.angle(q[..., 0])
        else:
            p = deconv1d(p, self.kernel_p)
            q = deconv1d(q, self.kernel_q)
            p = self.root_p(p)
            q = self.root_q(q)
            p = torch.angle(p[..., 0::2])
            q = torch.angle(q[..., 0::2])
            w, _ = torch.sort(torch.cat((p, q), dim=-1))

        w = w.view_as(a)
        w = self.convert(w)
        w = torch.cat((K, w), dim=-1)
        return w
