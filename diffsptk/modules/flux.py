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


class Flux(nn.Module):
    """Flux calculation module.

    Parameters
    ----------
    lag : int
        Lag of the distance calculation, :math:`L`.

    norm : int or float
        Order of norm.

    reduction : ['none', 'mean', 'batchmean', 'sum']
        Reduction type.

    """

    def __init__(self, lag=1, norm=2, reduction="mean"):
        super().__init__()

        assert reduction in ("none", "mean", "batchmean", "sum")

        self.lag = lag
        self.norm = norm
        self.reduction = reduction

    def forward(self, x, y=None):
        """Calculate flux, which is the distance between adjacent frames.

        Parameters
        ----------
        x : Tensor [shape=(..., N, D)]
            Input.

        y : Tensor [shape=(..., N, D)] or None
            Target (optional).

        Returns
        -------
        out : Tensor [shape=(..., N-\\|L\\|) or scalar]
            Flux.

        Examples
        --------
        >>> x = diffsptk.ramp(5).view(3, 2)
        >>> x
        tensor([[0., 1.],
                [2., 3.],
                [4., 5.]])
        >>> flux = diffsptk.Flux(reduction="none", norm=1)
        >>> f = flux(x)
        >>> f
        tensor([4., 4.])

        """
        return self._forward(x, y, self.lag, self.norm, self.reduction)

    @staticmethod
    def _forward(x, y, lag, norm, reduction):
        if y is None:
            y = x

        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (N,) -> (N, 1)
            y = y.unsqueeze(-1)

        if 0 < lag:
            diff = x[..., lag:, :] - y[..., :-lag, :]
        elif lag < 0:
            diff = y[..., -lag:, :] - x[..., :lag, :]
        else:
            diff = x - y
        flux = torch.linalg.vector_norm(diff, ord=norm, dim=-1)

        if reduction == "none":
            pass
        elif reduction == "sum":
            flux = flux.sum()
        elif reduction == "mean":
            flux = flux.mean() / x.size(-1) ** (1 / norm)
        elif reduction == "batchmean":
            flux = flux.mean()
        else:
            raise ValueError(f"reduction {reduction} is not supported.")
        return flux

    _func = _forward
