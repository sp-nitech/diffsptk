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

from ..misc.utils import check_size
from ..misc.utils import numpy_to_torch
from ..misc.utils import to_3d
from .pol_root import RootsToPolynomial


class LineSpectralPairsToLinearPredictiveCoefficients(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lsp2lpc.html>`_
    for details.

    Parameters
    ----------
    lpc_order : int >= 0 [scalar]
        Order of LPC, :math:`M`.

    log_gain : bool [scalar]
        If True, assume input gain is in log scale.

    """

    def __init__(self, lpc_order, log_gain=False):
        super(LineSpectralPairsToLinearPredictiveCoefficients, self).__init__()

        self.lpc_order = lpc_order
        self.log_gain = log_gain

        assert 0 <= self.lpc_order

        if self.lpc_order == 0:
            pass
        elif self.lpc_order == 1:
            self.pol_q = RootsToPolynomial(lpc_order + 1)
        elif self.lpc_order % 2 == 0:
            self.pol_p = RootsToPolynomial(lpc_order)
            self.pol_q = RootsToPolynomial(lpc_order)
            self.register_buffer("kernel_p", numpy_to_torch([[[-1, 1]]]))
            self.register_buffer("kernel_q", numpy_to_torch([[[1, 1]]]))
        else:
            self.pol_p = RootsToPolynomial(lpc_order - 1)
            self.pol_q = RootsToPolynomial(lpc_order + 1)
            self.register_buffer("kernel_p", numpy_to_torch([[[-1, 0, 1]]]))
            self.register_buffer("kernel_q", numpy_to_torch([[[0, 1, 0]]]))

    def forward(self, w):
        """Convert LSP to LPC.

        Parameters
        ----------
        w : Tensor [shape=(..., M+1)]
            LSP coefficients in radians.

        Returns
        -------
        a : Tensor [shape=(..., M+1)]
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

        K, w = torch.split(w, [1, self.lpc_order], dim=-1)

        if self.log_gain:
            K = torch.exp(K)
        if self.lpc_order == 0:
            return K

        z = torch.exp(1j * to_3d(w))
        p = z[..., 1::2]
        q = z[..., 0::2]
        if self.lpc_order == 1:
            q = self.pol_q(torch.cat([q, q.conj()], dim=-1)).real
            a = 0.5 * q[..., 1:-1]
        else:
            p = self.pol_p(torch.cat([p, p.conj()], dim=-1)).real
            q = self.pol_q(torch.cat([q, q.conj()], dim=-1)).real
            p = F.conv1d(p, self.kernel_p, padding=1 if self.lpc_order % 2 == 1 else 0)
            q = F.conv1d(q, self.kernel_q)
            a = 0.5 * (p + q)

        a = a.view_as(w)
        a = torch.cat((K, a), dim=-1)
        return a
