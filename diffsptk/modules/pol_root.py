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
import torch.nn.functional as F

from ..typing import Callable, Precomputed
from ..utils.private import check_size, filter_values
from .base import BaseFunctionalModule


class RootsToPolynomial(BaseFunctionalModule):
    """This is the opposite module to :func:`~diffsptk.PolynomialToRoots`.

    Parameters
    ----------
    order : int >= 1
        The order of the polynomial.

    eps : float >= 0 or None
        If the absolute values of the imaginary parts of the polynomial coefficients are
        all less than this value, they are considered as real numbers.

    in_format : ['rectangular', 'polar']
        The input format.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        order: int,
        *,
        eps: float | None = None,
        in_format: str | int = "rectangular",
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = order

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert roots to polynomial coefficients.

        Parameters
        ----------
        x : Tensor [shape=(..., M)]
            The roots, can be complex.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The polynomial coefficients.

        Examples
        --------
        >>> x = torch.tensor([3, 4, -1])
        >>> pol_root = diffsptk.RootsToPolynomial(x.size(-1))
        >>> a = pol_root(x)
        >>> a
        tensor([ 1, -6,  5, 12])

        """
        check_size(x.size(-1), self.in_dim, "number of roots")
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = RootsToPolynomial._precompute(x.size(-1), *args, **kwargs)
        return RootsToPolynomial._forward(x, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(order: int, eps: float | None) -> None:
        if order <= 0:
            raise ValueError("order must be positive.")
        if eps is not None and eps < 0:
            raise ValueError("eps must be non-negative.")

    @staticmethod
    def _precompute(
        order: int,
        eps: float | None = None,
        in_format: str | int = "rectangular",
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        RootsToPolynomial._check(order, eps)

        if eps is None:
            if dtype is None:
                dtype = torch.get_default_dtype()
            eps = 1e-4 if dtype == torch.float else 1e-8
        if in_format in (0, "rectangular"):
            formatter = lambda x: x
        elif in_format in (1, "polar"):
            formatter = lambda x: torch.polar(x.real, x.imag)
        else:
            raise ValueError(f"in_format {in_format} is not supported.")

        return (eps, formatter)

    @staticmethod
    def _forward(x: torch.Tensor, eps: float, formatter: Callable) -> torch.Tensor:
        x = formatter(x)
        M = x.size(-1)
        a = F.pad(torch.zeros_like(x), (1, 0), value=1)
        for m in range(M):
            z = a.clone()
            a[..., 1:] = z[..., 1:] - x[..., m : m + 1] * z[..., :-1]
        if torch.is_complex(a) and torch.all(a.imag.abs() < eps):
            a = a.real
        return a
