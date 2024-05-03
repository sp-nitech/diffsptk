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

from ..misc.utils import check_size
from ..misc.utils import to_3d
from .pol_root import RootsToPolynomial


class LineSpectralPairsToLinearPredictiveCoefficients(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lsp2lpc.html>`_
    for details.

    Parameters
    ----------
    lpc_order : int >= 0
        Order of LPC, :math:`M`.

    log_gain : bool
        If True, assume input gain is in log scale.

    """

    def __init__(self, lpc_order, log_gain=False):
        super().__init__()

        assert 0 <= lpc_order

        self.lpc_order = lpc_order
        self.log_gain = log_gain
        kernel_p, kernel_q = self._precompute(self.lpc_order)
        self.register_buffer("kernel_p", kernel_p)
        self.register_buffer("kernel_q", kernel_q)

    def forward(self, w):
        """Convert LSP to LPC.

        Parameters
        ----------
        w : Tensor [shape=(..., M+1)]
            LSP frequencies in radians.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            LPC coefficients.

        Examples
        --------
        >>> w = diffsptk.ramp(3)
        >>> lsp2lpc = diffsptk.LineSpectralPairsToLinearPredictiveCoefficients(3)
        >>> a = lsp2lpc(w)
        >>> a
        tensor([ 0.0000,  0.8658, -0.0698,  0.0335])

        """
        check_size(w.size(-1), self.lpc_order + 1, "dimension of LSP")
        return self._forward(w, self.log_gain, self.kernel_p, self.kernel_q)

    @staticmethod
    def _forward(w, log_gain, kernel_p, kernel_q):
        M = w.size(-1) - 1
        K, w = torch.split(w, [1, M], dim=-1)

        if log_gain:
            K = torch.exp(K)
        if M == 0:
            return K

        z = torch.exp(1j * to_3d(w))
        p = z[..., 1::2]
        q = z[..., 0::2]
        if M == 1:
            q = RootsToPolynomial._func(torch.cat([q, q.conj()], dim=-1), real=True)
            a = 0.5 * q[..., 1:-1]
        else:
            p = RootsToPolynomial._func(torch.cat([p, p.conj()], dim=-1), real=True)
            q = RootsToPolynomial._func(torch.cat([q, q.conj()], dim=-1), real=True)
            p = F.conv1d(p, kernel_p, padding=1 if M % 2 == 1 else 0)
            q = F.conv1d(q, kernel_q)
            a = 0.5 * (p + q)

        a = a.view_as(w)
        a = torch.cat((K, a), dim=-1)
        return a

    @staticmethod
    def _func(w, log_gain):
        kernels = LineSpectralPairsToLinearPredictiveCoefficients._precompute(
            w.size(-1) - 1, dtype=w.dtype, device=w.device
        )
        return LineSpectralPairsToLinearPredictiveCoefficients._forward(
            w, log_gain, *kernels
        )

    @staticmethod
    def _precompute(lpc_order, dtype=None, device=None):
        if lpc_order % 2 == 0:
            kernel_p = torch.tensor([-1.0, 1.0], dtype=dtype, device=device)
            kernel_q = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
        else:
            kernel_p = torch.tensor([-1.0, 0.0, 1.0], dtype=dtype, device=device)
            kernel_q = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        return kernel_p.view(1, 1, -1), kernel_q.view(1, 1, -1)
