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
from ..misc.utils import symmetric_toeplitz


class LevinsonDurbin(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/levdur.html>`_
    for details. The implementation is based on a simple matrix inversion.

    Parameters
    ----------
    lpc_order : int >= 0 [scalar]
        Order of LPC coefficients, :math:`M`.

    """

    def __init__(self, lpc_order):
        super(LevinsonDurbin, self).__init__()

        self.lpc_order = lpc_order

        assert 0 <= self.lpc_order

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
        >>> x = diffsptk.nrand(4)
        tensor([ 0.8226, -0.0284, -0.5715,  0.2127,  0.1217])
        >>> acorr = diffsptk.AutocorrelationAnalysis(2, 5)
        >>> levdur = diffsptk.LevinsonDurbin(2)
        >>> a = levdur(acorr(x))
        >>> a
        tensor([0.8726, 0.1475, 0.5270])

        """
        check_size(r.size(-1), self.lpc_order + 1, "dimension of autocorrelation")

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
