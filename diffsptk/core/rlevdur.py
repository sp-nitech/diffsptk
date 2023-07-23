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


class ReverseLevinsonDurbin(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/rlevdur.html>`_
    for details.

    Parameters
    ----------
    lpc_order : int >= 0 [scalar]
        Order of LPC coefficients, :math:`M`.

    """

    def __init__(self, lpc_order):
        super(ReverseLevinsonDurbin, self).__init__()

        self.lpc_order = lpc_order

        assert 0 <= self.lpc_order

        self.register_buffer("eye", torch.eye(self.lpc_order + 1))

    def forward(self, a):
        """Solve a Yule-Walker linear system given LPC coefficients.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            Gain and LPC coefficients.

        Returns
        -------
        r : Tensor [shape=(..., M+1)]
            Autocorrelation.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> acorr = diffsptk.AutocorrelationAnalysis(2, 5)
        >>> levdur = diffsptk.LevinsonDurbin(2)
        >>> rlevdur = diffsptk.ReverseLevinsonDurbin(2)
        >>> r = acorr(x)
        >>> r
        tensor([ 5.8784,  0.8978, -2.0951])
        >>> r2 = rlevdur(levdur(r))
        >>> r2
        tensor([ 5.8784,  0.8978, -2.0951])

        """
        check_size(a.size(-1), self.lpc_order + 1, "dimension of LPC coefficients")

        K, a1 = torch.split(a, [1, self.lpc_order], dim=-1)
        u = F.pad(a1.flip(-1), (0, 1), value=1)
        e = K**2

        U = [u]
        E = [e]
        for m in range(self.lpc_order):
            u0 = U[-1][..., :1]
            u1 = U[-1][..., 1 : self.lpc_order - m]
            t = 1 / (1 - u0**2)
            u = (u1 - u0 * u1.flip(-1)) * t
            u = F.pad(u, (0, m + 2))
            e = E[-1] * t
            U.append(u)
            E.append(e)
        U = torch.stack(U[::-1], dim=-1)
        E = torch.stack(E[::-1], dim=-1)

        V = torch.linalg.solve_triangular(U, self.eye, upper=True, unitriangular=True)
        r = torch.matmul(V[..., :1].mT * E, V).squeeze(-2)
        assert a.shape == r.shape
        return r
