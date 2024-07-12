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
from torch import nn

from ..misc.utils import check_size


class LineSpectralPairsStabilityCheck(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lspcheck.html>`_
    for details.

    Parameters
    ----------
    lsp_order : int >= 0
        Order of LSP, :math:`M`.

    rate : float in [0, 1]
        Rate of distance between two adjacent LSPs.

    n_iter : int >= 0
        Number of iterations for modification.

    warn_type : ['ignore', 'warn', 'exit']
        Warning type.

    """

    def __init__(self, lsp_order, rate=0, n_iter=1, warn_type="warn"):
        super().__init__()

        assert 0 <= lsp_order
        assert 0 <= rate <= 1
        assert 0 <= n_iter
        assert warn_type in ("ignore", "warn", "exit")

        self.lsp_order = lsp_order
        self.min_distance = self._precompute(lsp_order, rate)
        self.n_iter = n_iter
        self.warn_type = warn_type

    def forward(self, w):
        """Check stability of LSP.

        Parameters
        ----------
        w : Tensor [shape=(..., M+1)]
            LSP coefficients in radians.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            Modified LSP coefficients.

        Examples
        --------
        >>> w1 = torch.tensor([0, 0, 1]) * torch.pi
        >>> lspcheck = diffsptk.LineSpectralPairsStabilityCheck(2, rate=0.01)
        >>> w2 = lspcheck(w1)
        >>> w2
        tensor([0.0000, 0.0105, 3.1311])

        """
        check_size(w.size(-1), self.lsp_order + 1, "dimension of LSP")
        return self._forward(w, self.min_distance, self.n_iter, self.warn_type)

    @staticmethod
    def _forward(w, min_distance, n_iter, warn_type):
        K, w1 = torch.split(w, [1, w.size(-1) - 1], dim=-1)

        distance = w1[..., 1:] - w1[..., :-1]
        if torch.any(distance <= 0) or torch.any(w <= 0) or torch.any(torch.pi <= w):
            if warn_type == "ignore":
                pass
            elif warn_type == "warn":
                warnings.warn("Detected unstable LSP coefficients.")
            elif warn_type == "exit":
                raise RuntimeError("Detected unstable LSP coefficients.")
            else:
                raise RuntimeError

        w1 = w1.clone()
        for _ in range(n_iter):
            for m in range(w1.size(-1) - 1):
                n = m + 1
                distance = w1[..., n] - w1[..., m]
                step_size = 0.5 * torch.clip(min_distance - distance, min=0)
                w1[..., m] -= step_size
                w1[..., n] += step_size
            w1 = torch.clip(w1, min=min_distance, max=torch.pi - min_distance)
            distance = w1[..., 1:] - w1[..., :-1]
            if torch.all(min_distance - 1e-16 <= distance):
                break

        w2 = torch.cat((K, w1), dim=-1)
        return w2

    @staticmethod
    def _func(w, rate, n_iter, warn_type):
        const = LineSpectralPairsStabilityCheck._precompute(w.size(-1) - 1, rate)
        return LineSpectralPairsStabilityCheck._forward(w, const, n_iter, warn_type)

    @staticmethod
    def _precompute(lsp_order, rate):
        return rate * torch.pi / (lsp_order + 1)
