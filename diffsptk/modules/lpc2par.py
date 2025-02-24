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

from ..misc.utils import check_size
from ..misc.utils import get_gamma
from .base import BaseFunctionalModule


class LinearPredictiveCoefficientsToParcorCoefficients(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lpc2par.html>`_
    for details.

    Parameters
    ----------
    lpc_order : int >= 0
        The order of the LPC, :math:`M`.

    gamma : float in [-1, 1]
        The gamma parameter, :math:`\\gamma`.

    c : int >= 1 or None
        The number of filter stages.

    """

    def __init__(self, lpc_order, gamma=1, c=None):
        super().__init__()

        self.input_dim = lpc_order + 1

        self.values = LinearPredictiveCoefficientsToParcorCoefficients._precompute(
            lpc_order, gamma, c
        )

    def forward(self, a):
        """Convert LPC to PARCOR.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            The LPC coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The PARCOR coefficients.

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
        check_size(a.size(-1), self.input_dim, "dimension of LPC")
        return self._forward(a, *self.values)

    @staticmethod
    def _func(a, *args, **kwargs):
        values = LinearPredictiveCoefficientsToParcorCoefficients._precompute(
            a.size(-1) - 1, *args, **kwargs
        )
        return LinearPredictiveCoefficientsToParcorCoefficients._forward(a, *values)

    @staticmethod
    def _check(lpc_order, gamma, c):
        if lpc_order < 0:
            raise ValueError("lpc_order must be non-negative.")
        if 1 < abs(gamma):
            raise ValueError("gamma must be in [-1, 1].")
        if c is not None and c < 1:
            raise ValueError("c must be greater than or equal to 1.")

    @staticmethod
    def _precompute(lpc_order, gamma, c):
        LinearPredictiveCoefficientsToParcorCoefficients._check(lpc_order, gamma, c)
        return (get_gamma(gamma, c),)

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
