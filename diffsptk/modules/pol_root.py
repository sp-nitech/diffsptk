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
import torch.nn.functional as F

from ..misc.utils import check_size


class RootsToPolynomial(nn.Module):
    """This is the opposite module to :func:`~diffsptk.PolynomialToRoots`.

    order : int >= 1 [scalar]
        Order of coefficients.

    """

    def __init__(self, order):
        super(RootsToPolynomial, self).__init__()

        self.order = order

        assert 1 <= self.order

    def forward(self, x):
        """Convert roots to polynomial coefficients.

        Parameters
        ----------
        x : Tensor [shape=(..., M)]
            Complex roots.

        Returns
        -------
        a : Tensor [shape=(..., M+1)]
            Polynomial coefficients.

        Examples
        --------
        >>> x = torch.tensor([3, 4, -1])
        >>> pol_root = diffsptk.RootsToPolynomial(x.size(-1))
        >>> a = pol_root(x)
        >>> a
        tensor([ 1, -6,  5, 12])

        """
        check_size(x.size(-1), self.order, "number of roots")

        a = F.pad(torch.ones_like(x[..., :1]), (0, self.order))
        for m in range(self.order):
            a[..., 1:] = a[..., 1:].clone() - x[..., m : m + 1] * a[..., :-1].clone()
        return a
