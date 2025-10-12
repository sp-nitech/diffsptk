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
from ..utils.private import check_size, filter_values
from .base import BaseFunctionalModule


class AllPoleToAllZeroDigitalFilterCoefficients(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/norm0.html>`_
    for details.

    Parameters
    ----------
    filter_order : int >= 0
        The order of the filter coefficients, :math:`M`.

    """

    def __init__(self, filter_order: int) -> None:
        super().__init__()

        self.in_dim = filter_order + 1

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """Convert all-pole to all-zero filter coefficients vice versa.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            The all-pole or all-zero filter coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The all-zero or all-pole filter coefficients.

        Examples
        --------
        >>> import diffsptk
        >>> norm0 = diffsptk.AllPoleToAllZeroDigitalFilterCoefficients(3)
        >>> a = diffsptk.ramp(4, 1, -1)
        >>> a
        tensor([4., 3., 2., 1.])
        >>> b = norm0(a)
        >>> b
        tensor([0.2500, 0.7500, 0.5000, 0.2500])

        """
        check_size(a.size(-1), self.in_dim, "dimension of coefficients")
        return self._forward(a)

    @staticmethod
    def _func(a: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        AllPoleToAllZeroDigitalFilterCoefficients._precompute(
            a.size(-1) - 1, *args, **kwargs
        )
        return AllPoleToAllZeroDigitalFilterCoefficients._forward(a)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(filter_order: int) -> None:
        if filter_order < 0:
            raise ValueError("filter_order must be non-negative.")

    @staticmethod
    def _precompute(filter_order: int) -> Precomputed:
        AllPoleToAllZeroDigitalFilterCoefficients._check(filter_order)
        return (None,)

    @staticmethod
    def _forward(a: torch.Tensor) -> torch.Tensor:
        K, a1 = torch.split(a, [1, a.size(-1) - 1], dim=-1)
        b0 = torch.reciprocal(K)
        b1 = a1 * b0
        b = torch.cat((b0, b1), dim=-1)
        return b
