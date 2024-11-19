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

from scipy.special import comb
import torch
from torch import nn
import torch.nn.functional as F

from ..misc.utils import check_size
from ..misc.utils import hankel
from ..misc.utils import to
from ..misc.utils import vander
from .root_pol import PolynomialToRoots


class AutocorrelationToCompositeSinusoidalModelCoefficients(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/acr2csm.html>`_
    for details.

    Parameters
    ----------
    csm_order : int >= 0
        Order of CSM coefficients, :math:`M`.

    References
    ----------
    .. [1] S. Sagayama et al., "Duality theory of composite sinusoidal modeling and
           linear prediction," *Proceedings of ICASSP*, pp. 1261-1264, 1986.

    """

    def __init__(self, csm_order):
        super().__init__()

        assert 0 <= csm_order <= 30  # This is due to computational accuracy.
        assert csm_order % 2 == 1

        self.csm_order = csm_order
        self.register_buffer("C", self._precompute(self.csm_order))

    def forward(self, r):
        """Convert autocorrelation to CSM coefficients.

        Parameters
        ----------
        r : Tensor [shape=(..., M+1)]
            Autocorrelation.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            Composite sinusoidal model coefficients.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([ 0.0165, -2.3693,  0.1375, -0.2262,  1.3307])
        >>> acorr = diffsptk.Autocorrelation(5, 3)
        >>> acr2csm = diffsptk.AutocorrelationToCompositeSinusoidalModelCoefficients(3)
        >>> c = acr2csm(acorr(x))
        >>> c
        tensor([0.9028, 2.5877, 3.8392, 3.6153])

        """
        check_size(r.size(-1), self.csm_order + 1, "dimension of autocorrelation")
        return self._forward(r, self.C)

    @staticmethod
    def _forward(r, C):
        assert r.dtype == torch.double
        u = torch.matmul(r, C)
        u1, u2 = torch.tensor_split(u, 2, dim=-1)

        U = hankel(-u)
        p = torch.matmul(U.inverse(), u2.unsqueeze(-1)).squeeze(-1)
        x = PolynomialToRoots._func(F.pad(p.flip(-1), (1, 0), value=1))
        x, _ = torch.sort(x.real, descending=True)
        w = torch.acos(x)

        V = vander(x)
        m = torch.matmul(V.inverse(), u1.unsqueeze(-1)).squeeze(-1)
        c = torch.cat((w, m), dim=-1)
        return c

    @staticmethod
    def _func(r):
        C = AutocorrelationToCompositeSinusoidalModelCoefficients._precompute(
            r.size(-1) - 1, dtype=r.dtype, device=r.device
        )
        return AutocorrelationToCompositeSinusoidalModelCoefficients._forward(r, C)

    @staticmethod
    def _precompute(csm_order, dtype=None, device=None):
        N = csm_order + 1
        B = torch.zeros((N, N), dtype=torch.double, device=device)
        for n in range(N):
            z = 2**-n
            for k in range(n + 1):
                B[k, n] = comb(n, k, exact=True) * z

        C = torch.zeros((N, N), dtype=torch.double, device=device)
        for k in range(N):
            bias = k % 2
            center = k // 2
            length = center + 1
            C[bias : bias + 2 * length : 2, k] = B[
                bias + center : bias + center + length, k
            ]
        C[1:] *= 2
        return to(C, dtype=dtype)
