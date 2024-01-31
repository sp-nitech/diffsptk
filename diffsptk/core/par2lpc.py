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

import torch.nn as nn

from ..misc.utils import check_size


class ParcorCoefficientsToLinearPredictiveCoefficients(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/par2lpc.html>`_
    for details. This module may be slow due to recursive computation.

    Parameters
    ----------
    lpc_order : int >= 0 [scalar]
        Order of LPC, :math:`M`.

    """

    def __init__(self, lpc_order):
        super(ParcorCoefficientsToLinearPredictiveCoefficients, self).__init__()

        self.lpc_order = lpc_order

        assert 0 <= self.lpc_order

    def forward(self, k):
        """Convert PARCOR to LPC.

        Parameters
        ----------
        k : Tensor [shape=(..., M+1)]
            PARCOR coefficients.

        Returns
        -------
        a : Tensor [shape=(..., M+1)]
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

        a = k.clone()
        for m in range(2, self.lpc_order + 1):
            km = k[..., m : m + 1]
            am = a[..., 1:m]
            a[..., 1:m] = am + km * am.flip(-1)
        return a
