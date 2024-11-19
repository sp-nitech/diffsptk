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

from ..misc.utils import check_size
from ..misc.utils import to


class CompositeSinusoidalModelCoefficientsToAutocorrelation(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/csm2acr.html>`_
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

        assert 1 <= csm_order
        assert csm_order % 2 == 1

        self.csm_order = csm_order
        self.register_buffer("ramp", self._precompute(self.csm_order))

    def forward(self, c):
        """Convert CSM coefficients to autocorrelation.

        Parameters
        ----------
        c : Tensor [shape=(..., M+1)]
            Composite sinusoidal model coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            Autocorrelation.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> acorr = diffsptk.Autocorrelation(5, 3)
        >>> acr2csm = diffsptk.AutocorrelationToCompositeSinusoidalModelCoefficients(3)
        >>> csm2acr = diffsptk.CompositeSinusoidalModelCoefficientsToAutocorrelation(3)
        >>> r = acorr(x)
        >>> r
        tensor([ 8.8894, -0.1102, -4.1748,  0.7501])
        >>> r2 = csm2acr(acr2csm(r))
        >>> r2
        tensor([ 8.8894, -0.1102, -4.1748,  0.7501])

        """
        check_size(c.size(-1), self.csm_order + 1, "dimension of autocorrelation")
        return self._forward(c, self.ramp)

    @staticmethod
    def _forward(c, ramp):
        w, m = torch.tensor_split(c, 2, dim=-1)
        a = m.unsqueeze(-2)
        b = torch.cos(w.unsqueeze(-1) * ramp)
        r = torch.matmul(a, b).squeeze(-2)
        return r

    @staticmethod
    def _func(c):
        ramp = CompositeSinusoidalModelCoefficientsToAutocorrelation._precompute(
            c.size(-1) - 1, dtype=c.dtype, device=c.device
        )
        return CompositeSinusoidalModelCoefficientsToAutocorrelation._forward(c, ramp)

    @staticmethod
    def _precompute(csm_order, dtype=None, device=None):
        ramp = torch.arange(csm_order + 1, device=device)
        return to(ramp, dtype=dtype)
