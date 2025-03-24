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

import math

import torch

from ..typing import Precomputed
from ..utils.private import get_values
from .base import BaseFunctionalModule


class CepstralDistance(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/cdist.html>`_
    for details.

    Parameters
    ----------
    full : bool
        If True, include the constant term in the distance calculation.

    reduction : ['none', 'mean', 'batchmean', 'sum']
        The reduction type.

    References
    ----------
    .. [1] R. F. Kubichek, "Mel-cepstral distance measure for objective speech quality
           assessment," *Proceedings of IEEE Pacific Rim Conference on Communications
           Computers and Signal Processing*, vol. 1, pp. 125-128, 1993.

    """

    def __init__(self, full: bool = False, reduction: str = "mean") -> None:
        super().__init__()

        self.values = self._precompute(*get_values(locals()))

    def forward(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        """Calculate the cepstral distance between two inputs.

        Parameters
        ----------
        c1 : Tensor [shape=(..., M+1)]
            The input cepstral coefficients.

        c2 : Tensor [shape=(..., M+1)]
            The target cepstral coefficients.

        Returns
        -------
        out : Tensor [shape=(...,) or scalar]
            The cepstral distance.

        Examples
        --------
        >>> c1 = diffsptk.nrand(2, 2)
        tensor([[ 0.4296,  1.6517, -0.6022],
                [-1.0464, -0.6088, -0.9274]])
        >>> c2 = diffsptk.nrand(2, 2)
        tensor([[ 1.6441, -0.6962, -0.2524],
                [ 0.9344,  0.3965,  1.1494]])
        >>> cdist = diffsptk.CepstralDistance()
        >>> distance = cdist(c1, c2)
        >>> distance
        tensor(1.6551)

        """
        return self._forward(c1, c2, *self.values)

    @staticmethod
    def _func(c1: torch.Tensor, c2: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = CepstralDistance._precompute(*args, **kwargs)
        return CepstralDistance._forward(c1, c2, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check() -> None:
        pass

    @staticmethod
    def _precompute(full: bool, reduction: str) -> Precomputed:
        CepstralDistance._check()
        const = 10 * math.sqrt(2) / math.log(10) if full else 1
        return (const, reduction)

    @staticmethod
    def _forward(
        c1: torch.Tensor, c2: torch.Tensor, const: float, reduction: str
    ) -> torch.Tensor:
        distance = torch.linalg.vector_norm(c1[..., 1:] - c2[..., 1:], dim=-1)

        if reduction == "none":
            pass
        elif reduction == "sum":
            distance = distance.sum()
        elif reduction == "mean":
            distance = distance.mean() / ((c1.size(-1) - 1) ** 0.5)
        elif reduction == "batchmean":
            distance = distance.mean()
        else:
            raise ValueError(f"reduction {reduction} is not supported.")

        return const * distance
