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

from ..misc.utils import is_in
from ..misc.utils import symmetric_toeplitz


class PseudoLevinsonDurbinRecursion(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/levdur.html>`_
    for details. Note that the current implementation does not use the Durbin's
    algorithm though the class name includes it.

    Parameters
    ----------
    out_format : ['K', 'a', 'Ka', 'K,a']
        `K` is gain and `a` is LPC coefficients. If this is `Ka`, the two output
        tensors are concatenated and return the tensor instead of the tuple.

    """

    def __init__(self, out_format="K,a"):
        super(PseudoLevinsonDurbinRecursion, self).__init__()

        self.out_format = out_format
        assert is_in(self.out_format, ["K", "a", "Ka", "K,a"])

    def forward(self, r):
        """Solve a Yule-Walker linear system.

        Parameters
        ----------
        r : Tensor [shape=(..., M+1)]
            Autocorrelation.

        Returns
        -------
        K : Tensor [shape=(..., 1)]
            Gain.

        a : Tensor [shape=(..., M)]
            LPC coefficients.

        Examples
        --------
        >>> x = torch.nrand(5)
        tensor([ 0.8226, -0.0284, -0.5715,  0.2127,  0.1217])
        >>> acorr = diffsptk.AutocorrelationAnalysis(2, 5)
        >>> levdur = diffsptk.LevinsonDurbinRecursion(out_format="K,a")
        >>> a = levdur(acorr(x))
        >>> a
        (tensor([0.8726]), tensor([0.1475, 0.5270]))

        """
        # Make Toeplitz matrix.
        R = symmetric_toeplitz(r[..., :-1])

        # Solve system.
        r1 = r[..., 1:]
        a = torch.einsum("...mn,...m->...n", R.inverse(), -r1)

        # Compute gain.
        if "K" in self.out_format:
            r0 = r[..., 0]
            K = torch.sqrt(torch.einsum("...m,...m->...", r1, a) + r0)
            K = K.unsqueeze(-1)

        if self.out_format == "K":
            return K
        elif self.out_format == "a":
            return a
        elif self.out_format == "Ka":
            return torch.cat((K, a), dim=-1)
        elif self.out_format == "K,a":
            return K, a
        else:
            raise RuntimeError
