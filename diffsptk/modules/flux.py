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

from ..typing import Precomputed
from .base import BaseFunctionalModule


class Flux(BaseFunctionalModule):
    """Flux calculation module.

    Parameters
    ----------
    lag : int
        The lag of the distance calculation, :math:`L`.

    norm : int or float
        The order of the norm.

    reduction : ['none', 'mean', 'batchmean', 'sum']
        The reduction type.

    """

    def __init__(
        self, lag: int = 1, norm: int | float = 2, reduction: str = "mean"
    ) -> None:
        super().__init__()

        self.values = self._precompute(lag, norm, reduction)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate the flux, which is the distance between adjacent frames.

        Parameters
        ----------
        x : Tensor [shape=(..., N, D)]
            The input.

        y : Tensor [shape=(..., N, D)] or None
            The target (optional).

        Returns
        -------
        out : Tensor [shape=(..., N-\\|L\\|) or scalar]
            The flux.

        Examples
        --------
        >>> import diffsptk
        >>> flux = diffsptk.Flux(reduction="none", norm=1)
        >>> x = diffsptk.ramp(5).view(3, 2)
        >>> x
        tensor([[0., 1.],
                [2., 3.],
                [4., 5.]])
        >>> f = flux(x)
        >>> f
        tensor([4., 4.])

        """
        return self._forward(x, y, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, y: torch.Tensor | None, *args, **kwargs) -> torch.Tensor:
        values = Flux._precompute(*args, **kwargs)
        return Flux._forward(x, y, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check() -> None:
        pass

    @staticmethod
    def _precompute(lag: int, norm: int | float, reduction: str) -> Precomputed:
        Flux._check()
        return (lag, norm, reduction)

    @staticmethod
    def _forward(
        x: torch.Tensor,
        y: torch.Tensor | None,
        lag: int,
        norm: int | float,
        reduction: str,
    ) -> torch.Tensor:
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
            flux = flux.mean() / (x.size(-1) ** (1 / norm))
        elif reduction == "batchmean":
            flux = flux.mean()
        else:
            raise ValueError(f"reduction {reduction} is not supported.")

        return flux
