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

import warnings

import torch
import torch.nn as nn

from ..misc.utils import check_size


class LinearPredictiveCoefficientsToParcorCoefficients(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lpc2par.html>`_
    for details. This module may be slow due to recursive computation.

    Parameters
    ----------
    lpc_order : int >= 0 [scalar]
        Order of LPC, :math:`M`.

    gamma : float [-1 <= float <= 1]
        Gamma, :math:`\\gamma`.

    warn_type : ['ignore', 'warn', 'exit']
        Behavior for unstable LPC.

    """

    def __init__(self, lpc_order, gamma=1, warn_type="ignore"):
        super(LinearPredictiveCoefficientsToParcorCoefficients, self).__init__()

        self.lpc_order = lpc_order
        self.gamma = gamma
        self.warn_type = warn_type

        assert 0 <= self.lpc_order
        assert abs(self.gamma) <= 1
        assert self.warn_type in ("ignore", "warn", "exit")

    def forward(self, a):
        """Convert LPC to PARCOR.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            LPC coefficients.

        Returns
        -------
        k : Tensor [shape=(..., M+1)]
            PARCOR coefficients.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        tensor([ 0.7829, -0.2028,  1.6912,  0.1454,  0.4861])
        >>> lpc = diffsptk.LPC(3, 5)
        >>> a = lpc(x)
        >>> a
        tensor([ 1.6036,  0.0573, -0.5615, -0.0638])
        >>> lpc2par = diffsptk.LinearPredictiveCoefficientsToParcorCoefficients(3)
        >>> k = lpc2par(a)
        >>> k
        tensor([ 1.6036,  0.0491, -0.5601, -0.0638])

        """
        check_size(a.size(-1), self.lpc_order + 1, "dimension of LPC")

        K, a = torch.split(a, [1, self.lpc_order], dim=-1)

        ks = []
        a = a * self.gamma
        for m in range(self.lpc_order - 1, -1, -1):
            km = a[..., m : m + 1]
            if torch.any(1 <= torch.abs(km)):
                if self.warn_type == "ignore":
                    pass
                elif self.warn_type == "warn":
                    warnings.warn("Unstable LPC coefficients")
                elif self.warn_type == "exit":
                    raise RuntimeError("Unstable LPC coefficients")
                else:
                    raise RuntimeError

            ks.append(km)
            z = 1 - km * km
            k = a[..., :-1]
            a = (k - km * k.flip(-1)) / z

        ks.append(K)
        k = torch.cat(ks[::-1], dim=-1)
        return k
