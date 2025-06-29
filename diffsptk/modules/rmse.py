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
from ..utils.private import filter_values
from .base import BaseFunctionalModule


class RootMeanSquareError(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/rmse.html>`_
    for details.

    Parameters
    ----------
    reduction : ['none', 'mean', 'sum']
        The reduction type.

    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate RMSE.

        Parameters
        ----------
        x : Tensor [shape=(..., D)]
            The input.

        y : Tensor [shape=(..., D)]
            The target.

        Returns
        -------
        out : Tensor [shape=(...,) or scalar]
            The RMSE.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([-0.5945, -0.2401,  0.8633, -0.6464,  0.4515])
        >>> y = diffsptk.nrand(4)
        >>> y
        tensor([-0.4025,  0.9367,  1.1299,  3.1986, -0.2832])
        >>> rmse = diffsptk.RootMeanSquaredError()
        >>> e = rmse(x, y)
        >>> e
        tensor(1.8340)

        """
        return self._forward(x, y, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = RootMeanSquareError._precompute(*args, **kwargs)
        return RootMeanSquareError._forward(x, y, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check() -> None:
        pass

    @staticmethod
    def _precompute(reduction: str) -> Precomputed:
        RootMeanSquareError._check()
        return (reduction,)

    @staticmethod
    def _forward(x: torch.Tensor, y: torch.Tensor, reduction: str) -> torch.Tensor:
        error = torch.linalg.vector_norm(x - y, dim=-1) / (x.size(-1) ** 0.5)

        if reduction == "none":
            pass
        elif reduction == "sum":
            error = error.sum()
        elif reduction == "mean":
            error = error.mean()
        else:
            raise ValueError(f"reduction {reduction} is not supported.")

        return error
