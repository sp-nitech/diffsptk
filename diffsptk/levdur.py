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


class PseudoLevinsonDurbinRecursion(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/levdur.html>`_
    for details. Note that the current implementation does not use the Durbin's
    algorithm though the class name includes it.
    """

    def __init__(self):
        super(PseudoLevinsonDurbinRecursion, self).__init__()

    def forward(self, r, n_out=2):
        """Solve a Yule-Walker linear system.

        Parameters
        ----------
        r : Tensor [shape=(..., M+1)]
            Autocorrelation.

        n_out : int [1 <= n_out <= 2]
            If `n_out` is 1, the two output tensors are concatenated and
            return the tensor instead of tuple.

        Returns
        -------
        g : Tensor [shape=(..., 1)]
            Gain.

        a : Tensor [shape=(..., M)]
            LPC coefficients.

        """
        M = r.size(-1) - 1

        # Make Toeplitz matrix.
        r_0M = r[..., :-1]
        r_M1 = r_0M[..., 1:].flip(-1)
        r_MM = torch.cat((r_M1, r_0M), dim=-1)
        R = r_MM.unfold(-1, M, 1).flip(-2)

        # Solve system.
        r_0 = r[..., 0]
        r_1L = r[..., 1:]
        a = torch.einsum("...mn,...m->...n", R.inverse(), -r_1L)

        # Compute gain.
        g = torch.sqrt(torch.einsum("...m,...m->...", r_1L, a) + r_0)
        g = g.unsqueeze(-1)

        if n_out == 1:
            return torch.cat((g, a), dim=-1)
        elif n_out == 2:
            return g, a
        else:
            ValueError
