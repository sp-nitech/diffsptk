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

from torch import nn

from ..misc.utils import check_size
from ..misc.utils import get_gamma


class ParcorCoefficientsToLinearPredictiveCoefficients(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/par2lpc.html>`_
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

    def forward(self, k):
        """Convert PARCOR to LPC.

        Parameters
        ----------
        k : Tensor [shape=(..., M+1)]
            PARCOR coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            LPC coefficients.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([ 0.7829, -0.2028,  1.6912,  0.1454,  0.4861])
        >>> lpc = diffsptk.LPC(3, 5)
        >>> a = lpc(x)
        >>> a
        tensor([ 1.6036,  0.0573, -0.5615, -0.0638])
        >>> lpc2par = diffsptk.LinearPredictiveCoefficientsToParcorCoefficients(3)
        >>> par2lpc = diffsptk.ParcorCoefficientsToLinearPredictiveCoefficients(3)
        >>> a2 = par2lpc(lpc2par(a))
        >>> a2
        tensor([ 1.6036,  0.0573, -0.5615, -0.0638])

        """
        check_size(k.size(-1), self.lpc_order + 1, "dimension of PARCOR")
        return self._forward(k, self.gamma)

    @staticmethod
    def _forward(k, gamma):
        a = k / gamma
        for m in range(2, k.size(-1)):
            km = k[..., m : m + 1]
            am = a[..., 1:m]
            a[..., 1:m] = am + km * am.flip(-1)
        return a

    @staticmethod
    def _func(k, gamma=1, c=None):
        gamma = ParcorCoefficientsToLinearPredictiveCoefficients._precompute(gamma, c)
        return ParcorCoefficientsToLinearPredictiveCoefficients._forward(k, gamma)

    @staticmethod
    def _precompute(gamma, c):
        return get_gamma(gamma, c)
