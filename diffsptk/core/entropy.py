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
import torch.nn as nn


class Entropy(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/entropy.html>`_
    for details.

    Parameters
    ----------
    unit : ['bit', 'nat', 'dit']
        Unit of entropy.

    """

    def __init__(self, unit="nat"):
        super(Entropy, self).__init__()

        if unit == 0 or unit == "bit":
            self.convert = lambda x: x * math.log2(math.e)
        elif unit == 1 or unit == "nat":
            self.convert = lambda x: x
        elif unit == 2 or unit == "dit":
            self.convert = lambda x: x * math.log10(math.e)
        else:
            raise ValueError(f"unit {unit} is not supported")

    def forward(self, p):
        """Compute entropy from probability sequence.

        Parameters
        ----------
        p : Tensor [shape=(..., N)]
            Probability sequence.

        Returns
        -------
        h : Tensor [shape=(...,)]
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
        h = torch.special.entr(p).sum(-1)
        h = self.convert(h)
        return h
