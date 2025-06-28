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
from ..utils.private import filter_values
from .base import BaseFunctionalModule


class Entropy(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/entropy.html>`_
    for details.

    Parameters
    ----------
    out_format : ['bit', 'nat', 'dit']
        The unit of the entropy.

    """

    def __init__(self, out_format: str | int = "nat") -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute entropy from probability sequence.

        Parameters
        ----------
        p : Tensor [shape=(..., N)]
            The probability sequence.

        Returns
        -------
        out : Tensor [shape=(...,)]
            The entropy.

        Examples
        --------
        >>> p = diffsptk.step(3) / 4
        >>> p
        tensor([0.2500, 0.2500, 0.2500, 0.2500])
        >>> entropy = diffsptk.Entropy("bit")
        >>> h = entropy(p)
        >>> h
        tensor(2.)

        """
        return self._forward(p, *self.values)

    @staticmethod
    def _func(p: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = Entropy._precompute(*args, **kwargs)
        return Entropy._forward(p, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check() -> None:
        pass

    @staticmethod
    def _precompute(out_format: str | int) -> Precomputed:
        Entropy._check()
        if out_format in (0, "bit"):
            c = math.log2(math.e)
        elif out_format in (1, "nat"):
            c = 1
        elif out_format in (2, "dit"):
            c = math.log10(math.e)
        else:
            raise ValueError(f"out_format {out_format} is not supported.")
        return (c,)

    @staticmethod
    def _forward(p: torch.Tensor, c: float) -> torch.Tensor:
        return c * torch.special.entr(p).sum(-1)
