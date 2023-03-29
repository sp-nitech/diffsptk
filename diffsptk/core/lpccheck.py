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

from .lpc2par import LinearPredictiveCoefficientsToParcorCoefficients
from .par2lpc import ParcorCoefficientsToLinearPredictiveCoefficients


class LinearPredictiveCoefficientsStabilityCheck(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lpccheck.html>`_
    for details.

    Parameters
    ----------
    lpc_order : int >= 0 [scalar]
        Order of LPC, :math:`M`.

    margin : [0 < float < 1]
        Margin.

    warn_type : ['ignore', 'warn', 'exit']
        Behavior for unstable LPC.

    """

    def __init__(self, lpc_order, margin=1e-16, warn_type="warn"):
        super(LinearPredictiveCoefficientsStabilityCheck, self).__init__()

        self.bound = 1 - margin

        assert 0 < margin and margin < 1

        self.lpc2par = LinearPredictiveCoefficientsToParcorCoefficients(
            lpc_order, warn_type=warn_type
        )
        self.par2lpc = ParcorCoefficientsToLinearPredictiveCoefficients(lpc_order)

    def forward(self, a1):
        """Check stability of LPC.

        Parameters
        ----------
        a1 : Tensor [shape=(..., M+1)]
            LPC coefficients.

        Returns
        -------
        a2 : Tensor [shape=(..., M+1)]
            Modified LPC coefficients.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        tensor([-0.9966, -0.2970, -0.2173,  0.0594,  0.5831])
        >>> lpc = diffsptk.LPC(3, 5)
        >>> a = lpc(x)
        >>> a
        tensor([ 1.1528, -0.2613, -0.0274,  0.1778])
        >>> lpccheck = diffsptk.LinearPredictiveCoefficientsStabilityCheck(3)
        >>> a2 = lpccheck(a)
        tensor([ 1.1528, -0.2613, -0.0274,  0.1778])

        """
        k1 = self.lpc2par(a1)
        K, k = torch.split(k1, [1, k1.size(-1) - 1], dim=-1)
        k = torch.clip(k, -self.bound, self.bound)
        k2 = torch.cat((K, k), dim=-1)
        a2 = self.par2lpc(k2)
        return a2
