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


class LineSpectralPairsStabilityCheck(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lspcheck.html>`_
    for details.

    Parameters
    ----------
    lsp_order : int >= 0 [scalar]
        Order of LSP, :math:`M`.

    rate : [0 <= float <= 1]
        Rate of distance between two adjacent LSPs.

    n_iter : int >= 0 [scalar]
        Number of iterations for modification.

    warn_type : ['ignore', 'warn', 'exit']
        Behavior for unstable LSP.

    """

    def __init__(self, lsp_order, rate=0, n_iter=1, warn_type="warn"):
        super(LineSpectralPairsStabilityCheck, self).__init__()

        self.lsp_order = lsp_order
        self.min_distance = rate * torch.pi / (self.lsp_order + 1)
        self.n_iter = n_iter
        self.warn_type = warn_type

        assert 0 <= self.lsp_order
        assert 0 <= rate <= 1
        assert 0 <= self.n_iter
        assert self.warn_type in ("ignore", "warn", "exit")

    def forward(self, w1):
        """Check stability of LSP.

        Parameters
        ----------
        w1 : Tensor [shape=(..., M+1)]
            LSP coefficients in radians.

        Returns
        -------
        w2 : Tensor [shape=(..., M+1)]
            Modified LSP coefficients.

        Examples
        --------
        >>> w1 = torch.tensor([0, 0, 1]) * torch.pi
        >>> lspcheck = diffsptk.LineSpectralPairsStabilityCheck(2, rate=0.01)
        >>> w2 = lspcheck(w1)
        >>> w2
        tensor([0.0000, 0.0105, 3.1311])

        """
        check_size(w1.size(-1), self.lsp_order + 1, "dimension of LSP")

        K, w = torch.split(w1, [1, self.lsp_order], dim=-1)
        distance = w[..., 1:] - w[..., :-1]
        if torch.any(distance <= 0) or torch.any(w <= 0) or torch.any(torch.pi <= w):
            if self.warn_type == "ignore":
                pass
            elif self.warn_type == "warn":
                warnings.warn("Unstable LSP coefficients")
            elif self.warn_type == "exit":
                raise RuntimeError("Unstable LSP coefficients")
            else:
                raise RuntimeError

        w = w.clone()
        for _ in range(self.n_iter):
            for m in range(self.lsp_order - 1):
                n = m + 1
                distance = w[..., n] - w[..., m]
                step_size = 0.5 * torch.clip(self.min_distance - distance, 0)
                w[..., m] -= step_size
                w[..., n] += step_size
            w = torch.clip(w, self.min_distance, torch.pi - self.min_distance)
            distance = w[..., 1:] - w[..., :-1]
            if torch.all(self.min_distance - 1e-16 <= distance):
                break

        w2 = torch.cat((K, w), dim=-1)
        return w2
