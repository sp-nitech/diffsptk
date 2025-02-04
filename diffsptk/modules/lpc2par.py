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
from ..misc.utils import get_gamma


class LinearPredictiveCoefficientsToParcorCoefficients(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lpc2par.html>`_
    for details.

    Parameters
    ----------
    lpc_order : int >= 0
        Order of LPC, :math:`M`.

    gamma : float in [-1, 1]
        Gamma, :math:`\\gamma`.

    c : int >= 1 or None
        Number of stages.

    """

    def __init__(self, lpc_order, gamma=1, c=None):
        super().__init__()

        assert 0 <= lpc_order
        assert abs(gamma) <= 1

        self.lpc_order = lpc_order
        self.gamma = self._precompute(gamma, c)

    def forward(self, a):
        """Convert LPC to PARCOR.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            LPC coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            PARCOR coefficients.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([ 0.7829, -0.2028,  1.6912,  0.1454,  0.4861])
        >>> lpc = diffsptk.LPC(5, 3)
        >>> a = lpc(x)
        >>> a
        tensor([ 1.6036,  0.0573, -0.5615, -0.0638])
        >>> lpc2par = diffsptk.LinearPredictiveCoefficientsToParcorCoefficients(3)
        >>> k = lpc2par(a)
        >>> k
        tensor([ 1.6036,  0.0491, -0.5601, -0.0638])

        """
        check_size(a.size(-1), self.lpc_order + 1, "dimension of LPC")
        return self._forward(a, self.gamma)

    @staticmethod
    def _forward(a, gamma):
        M = a.size(-1) - 1
        K, a = torch.split(a, [1, M], dim=-1)

        ks = []
        a = a * gamma
        for m in reversed(range(M)):
            km = a[..., m : m + 1]
            ks.append(km)
            if m == 0:
                break
            z = 1 - km * km
            k = a[..., :-1]
            a = (k - km * k.flip(-1)) / z

        ks.append(K)
        k = torch.cat(ks[::-1], dim=-1)
        return k

    @staticmethod
    def _func(a, gamma=1, c=None):
        gamma = LinearPredictiveCoefficientsToParcorCoefficients._precompute(gamma, c)
        return LinearPredictiveCoefficientsToParcorCoefficients._forward(a, gamma)

    @staticmethod
    def _precompute(gamma, c):
        return get_gamma(gamma, c)
