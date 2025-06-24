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

from ..typing import Precomputed
from ..utils.private import check_size, get_values, hankel, to, vander
from .base import BaseFunctionalModule
from .root_pol import PolynomialToRoots


class AutocorrelationToCompositeSinusoidalModelCoefficients(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/acr2csm.html>`_
    for details.

    Parameters
    ----------
    acr_order : int >= 0
        The order of the autocorrelation, :math:`M`.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] S. Sagayama et al., "Duality theory of composite sinusoidal modeling and
           linear prediction," *Proceedings of ICASSP*, pp. 1261-1264, 1986.

    """

    def __init__(
        self,
        acr_order: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = acr_order + 1

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("C", tensors[0])

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Convert autocorrelation to CSM coefficients.

        Parameters
        ----------
        r : Tensor [shape=(..., M+1)]
            The autocorrelation.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The CSM coefficients.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([ 0.0165, -2.3693,  0.1375, -0.2262,  1.3307])
        >>> acorr = diffsptk.Autocorrelation(5, 3)
        >>> acr2csm = diffsptk.AutocorrelationToCompositeSinusoidalModelCoefficients(3)
        >>> c = acr2csm(acorr(x))
        >>> c
        tensor([0.9028, 2.5877, 3.8392, 3.6153])

        """
        check_size(r.size(-1), self.in_dim, "dimension of autocorrelation")
        return self._forward(r, **self._buffers)

    @staticmethod
    def _func(r: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _, tensors = (
            AutocorrelationToCompositeSinusoidalModelCoefficients._precompute(
                r.size(-1) - 1, *args, **kwargs, device=r.device, dtype=r.dtype
            )
        )
        return AutocorrelationToCompositeSinusoidalModelCoefficients._forward(
            r, *tensors
        )

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(acr_order: int) -> None:
        if acr_order <= 0 or acr_order % 2 == 0:
            raise ValueError("acr_order must be a positive odd number.")
        if 30 < acr_order:
            raise ValueError("acr_order must be small due to computational accuracy.")

    @staticmethod
    def _precompute(
        acr_order: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        from scipy.special import comb

        AutocorrelationToCompositeSinusoidalModelCoefficients._check(acr_order)
        N = acr_order + 1
        B = torch.zeros((N, N), device=device, dtype=torch.double)
        for n in range(N):
            z = 2**-n
            for k in range(n + 1):
                B[k, n] = comb(n, k, exact=True) * z

        C = torch.zeros((N, N), device=device, dtype=torch.double)
        for k in range(N):
            bias = k % 2
            center = k // 2
            length = center + 1
            C[bias : bias + 2 * length : 2, k] = B[
                bias + center : bias + center + length, k
            ]
        C[1:] *= 2
        return None, None, (to(C, dtype=dtype),)

    @staticmethod
    def _forward(r: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        if r.dtype != torch.double or C.dtype != torch.double:
            raise ValueError("Only double precision is supported.")

        u = torch.matmul(r, C)
        u1, u2 = torch.tensor_split(u, 2, dim=-1)

        U = hankel(-u)
        p = torch.matmul(U.inverse(), u2.unsqueeze(-1)).squeeze(-1)
        x = PolynomialToRoots._func(F.pad(p.flip(-1), (1, 0), value=1))
        x, _ = torch.sort(x.real, descending=True)
        w = torch.acos(x)

        V = vander(x)
        m = torch.matmul(V.inverse(), u1.unsqueeze(-1)).squeeze(-1)
        c = torch.cat((w, m), dim=-1)
        return c
