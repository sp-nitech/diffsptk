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
from ..misc.utils import symmetric_toeplitz


class LevinsonDurbin(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/levdur.html>`_
    for details. The implementation is based on a simple matrix inversion.

    Parameters
    ----------
    lpc_order : int >= 0
        Order of LPC coefficients, :math:`M`.

    """

    def __init__(self, lpc_order):
        super().__init__()

        assert 0 <= lpc_order

        self.lpc_order = lpc_order

    def forward(self, r):
        """Solve a Yule-Walker linear system.

        Parameters
        ----------
        r : Tensor [shape=(..., M+1)]
            Autocorrelation.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            Gain and LPC coefficients.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        tensor([ 0.8226, -0.0284, -0.5715,  0.2127,  0.1217])
        >>> acorr = diffsptk.Autocorrelation(5, 2)
        >>> levdur = diffsptk.LevinsonDurbin(2)
        >>> a = levdur(acorr(x))
        >>> a
        tensor([0.8726, 0.1475, 0.5270])

        """
        check_size(r.size(-1), self.lpc_order + 1, "dimension of autocorrelation")
        return self._forward(r)

    @staticmethod
    def _forward(r):
        r0, r1 = torch.split(r, [1, r.size(-1) - 1], dim=-1)

        # Make Toeplitz matrix.
        R = symmetric_toeplitz(r[..., :-1])  # [..., M, M]

        # Solve system.
        a = torch.matmul(R.inverse(), -r1.unsqueeze(-1)).squeeze(-1)

        # Compute gain.
        K = torch.sqrt((r1 * a).sum(-1, keepdim=True) + r0)

        a = torch.cat((K, a), dim=-1)
        return a

    _func = _forward
