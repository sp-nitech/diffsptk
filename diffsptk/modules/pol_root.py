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
import torch.nn.functional as F

from ..misc.utils import check_size


class RootsToPolynomial(nn.Module):
    """This is the opposite module to :func:`~diffsptk.PolynomialToRoots`.

    Parameters
    ----------
    order : int >= 1
        Order of polynomial.

    """

    def __init__(self, order):
        super().__init__()

        assert 1 <= order

        self.order = order

    def forward(self, x, real=False):
        """Convert roots to polynomial coefficients.

        Parameters
        ----------
        x : Tensor [shape=(..., M)]
            Complex roots.

        real : bool
            If True, return as real numbers.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
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
        return self._forward(x, real)

    @staticmethod
    def _forward(x, real):
        M = x.size(-1)
        a = F.pad(torch.zeros_like(x), (1, 0), value=1)
        for m in range(M):
            z = a.clone()
            a[..., 1:] = z[..., 1:] - x[..., m : m + 1] * z[..., :-1]
        return a.real if real else a

    _func = _forward
