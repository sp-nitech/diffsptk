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

from ..misc.utils import symmetric_toeplitz


class PseudoLevinsonDurbinRecursion(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/levdur.html>`_
    for details. Note that the current implementation does not use the Durbin's
    algorithm though the class name includes it.
    """

    def __init__(self):
        super(PseudoLevinsonDurbinRecursion, self).__init__()

    def forward(self, r):
        """Solve a Yule-Walker linear system.

        Parameters
        ----------
        r : Tensor [shape=(..., M+1)]
            Autocorrelation.

        Returns
        -------
        a : Tensor [shape=(..., M+1)]
            Gain and LPC coefficients.

        Examples
        --------
        >>> x = torch.randn(5)
        tensor([ 0.8226, -0.0284, -0.5715,  0.2127,  0.1217])
        >>> acorr = diffsptk.AutocorrelationAnalysis(2, 5)
        >>> levdur = diffsptk.LevinsonDurbinRecursion()
        >>> a = levdur(acorr(x))
        >>> a
        tensor([0.8726, 0.1475, 0.5270])

        """
        # Make Toeplitz matrix.
        R = symmetric_toeplitz(r[..., :-1])

        # Solve system.
        r1 = r[..., 1:]
        a = torch.einsum("...mn,...m->...n", R.inverse(), -r1)

        # Compute gain.
        r0 = r[..., 0]
        K = torch.sqrt(torch.einsum("...m,...m->...", r1, a) + r0)
        K = K.unsqueeze(-1)

        a = torch.cat((K, a), dim=-1)
        return a
