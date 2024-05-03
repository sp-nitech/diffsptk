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
from torch import nn


class Entropy(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/entropy.html>`_
    for details.

    Parameters
    ----------
    out_format : ['bit', 'nat', 'dit']
        Unit of entropy.

    """

    def __init__(self, out_format="nat"):
        super().__init__()

        self.const = self._precompute(out_format)

    def forward(self, p):
        """Compute entropy from probability sequence.

        Parameters
        ----------
        p : Tensor [shape=(..., N)]
            Probability sequence.

        Returns
        -------
        out : Tensor [shape=(...,)]
            Entropy.

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
        return self._forward(p, self.const)

    @staticmethod
    def _forward(p, const):
        h = torch.special.entr(p).sum(-1) * const
        return h

    @staticmethod
    def _func(p, out_format):
        const = Entropy._precompute(out_format)
        return Entropy._forward(p, const)

    @staticmethod
    def _precompute(out_format):
        if out_format in (0, "bit"):
            return math.log2(math.e)
        elif out_format in (1, "nat"):
            return 1
        elif out_format in (2, "dit"):
            return math.log10(math.e)
        raise ValueError(f"out_format {out_format} is not supported.")
