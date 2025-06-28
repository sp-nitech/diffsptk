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
from ..utils.private import check_size, filter_values, symmetric_toeplitz
from .base import BaseFunctionalModule


class LevinsonDurbin(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/levdur.html>`_
    for details. The implementation is based on a simple matrix inversion.

    Parameters
    ----------
    lpc_order : int >= 0
        The order of the LPC coefficients, :math:`M`.

    eps : float >= 0
        A small value to improve numerical stability.

    """

    def __init__(self, lpc_order: int, eps: float = 0) -> None:
        super().__init__()

        self.in_dim = lpc_order + 1

        _, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("eye", tensors[0])

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Solve a Yule-Walker linear system.

        Parameters
        ----------
        r : Tensor [shape=(..., M+1)]
            The autocorrelation.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The gain and the LPC coefficients.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([ 0.8226, -0.0284, -0.5715,  0.2127,  0.1217])
        >>> acorr = diffsptk.Autocorrelation(5, 2)
        >>> levdur = diffsptk.LevinsonDurbin(2)
        >>> a = levdur(acorr(x))
        >>> a
        tensor([0.8726, 0.1475, 0.5270])

        """
        check_size(r.size(-1), self.in_dim, "dimension of autocorrelation")
        return self._forward(r, **self._buffers)

    @staticmethod
    def _func(r: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _, tensors = LevinsonDurbin._precompute(
            r.size(-1) - 1, *args, **kwargs, device=r.device, dtype=r.dtype
        )
        return LevinsonDurbin._forward(r, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(lpc_order: int, eps: float) -> None:
        if lpc_order < 0:
            raise ValueError("lpc_order must be non-negative.")
        if eps < 0:
            raise ValueError("eps must be non-negative.")

    @staticmethod
    def _precompute(
        lpc_order: int,
        eps: float = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        LevinsonDurbin._check(lpc_order, eps)
        eye = torch.eye(lpc_order, device=device, dtype=dtype) * eps
        return None, None, (eye,)

    @staticmethod
    def _forward(r: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
        r0, r1 = torch.split(r, [1, r.size(-1) - 1], dim=-1)

        # Make Toeplitz matrix.
        R = symmetric_toeplitz(r[..., :-1]) + eye  # [..., M, M]

        # Solve system.
        a = torch.matmul(R.inverse(), -r1.unsqueeze(-1)).squeeze(-1)

        # Compute gain.
        K = torch.sqrt((r1 * a).sum(-1, keepdim=True) + r0)

        a = torch.cat((K, a), dim=-1)
        return a
