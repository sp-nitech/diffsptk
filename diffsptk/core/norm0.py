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
import torch.nn as nn

from ..misc.utils import check_size


class AllPoleToAllZeroDigitalFilterCoefficients(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/norm0.html>`_
    for details.

    Parameters
    ----------
    filter_order : int >= 0 [scalar]
        Order of filter coefficients, :math:`M`.

    """

    def __init__(self, filter_order):
        super(AllPoleToAllZeroDigitalFilterCoefficients, self).__init__()

        self.filter_order = filter_order
        assert 0 <= self.filter_order

    def forward(self, a):
        """Convert all-pole to all-zero filter coefficients vice versa.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            All-pole or all-zero filter coefficients.

        Returns
        -------
        b : Tensor [shape=(..., M+1)]
            All-zero or all-pole filter coefficients.

        Examples
        --------
        >>> a = diffsptk.ramp(4, 1, -1)
        >>> norm0 = diffsptk.AllPoleToAllZeroDigitalFilterCoefficients(3)
        >>> b = norm0(a)
        >>> b
        tensor([0.2500, 0.7500, 0.5000, 0.2500])

        """
        check_size(a.size(-1), self.filter_order + 1, "dimension of coefficients")

        K, a1 = torch.split(a, [1, self.filter_order], dim=-1)
        b0 = torch.reciprocal(K)
        b1 = a1 * b0
        b = torch.cat((b0, b1), dim=-1)
        return b
