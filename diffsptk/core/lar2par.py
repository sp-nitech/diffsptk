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

from ..misc.utils import check_size


class LogAreaRatioToParcorCoefficients(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lar2par.html>`_
    for details.

    Parameters
    ----------
    par_order : int >= 0 [scalar]
        Order of PARCOR, :math:`M`.

    """

    def __init__(self, par_order):
        super(LogAreaRatioToParcorCoefficients, self).__init__()

        self.par_order = par_order

        assert 0 <= self.par_order

    def forward(self, A):
        """Convert LAR to PARCOR.

        Parameters
        ----------
        A : Tensor [shape=(..., M+1)]
            Log area ratio.

        Returns
        -------
        k : Tensor [shape=(..., M+1)]
            PARCOR coefficients.

        Examples
        --------
        >>> k = diffsptk.ramp(1, 4) * 0.1
        >>> par2lar = diffsptk.ParcorCoefficientsToLogAreaRatio(3)
        >>> A = par2lar(k)
        >>> A
        tensor([0.1000, 0.4055, 0.6190, 0.8473])

        """
        check_size(A.size(-1), self.par_order + 1, "dimension of parcor")

        K, A1 = torch.split(A, [1, self.par_order], dim=-1)
        k = torch.cat((K, torch.tanh(0.5 * A1)), dim=-1)
        return k
