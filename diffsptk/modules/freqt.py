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
from ..utils.private import check_size, get_values, to
from .base import BaseFunctionalModule


class FrequencyTransform(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/freqt.html>`_
    for details.

    Parameters
    ----------
    in_order : int >= 0
        The order of the input sequence, :math:`M_1`.

    out_order : int >= 0
        The order of the output sequence, :math:`M_2`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    References
    ----------
    .. [1] A. V. Oppenheim et al, "Discrete representation of signals," *Proceedings of
           the IEEE*, vol. 60, no. 6, pp. 681-691, 1972.

    """

    def __init__(self, in_order: int, out_order: int, alpha: float = 0) -> None:
        super().__init__()

        self.in_dim = in_order + 1

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("A", tensors[0])

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Perform frequency transform.

        Parameters
        ----------
        c : Tensor [shape=(..., M1+1)]
            The input cepstral coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M2+1)]
            The warped cepstral coefficients.

        Examples
        --------
        >>> c1 = diffsptk.ramp(3)
        >>> c1
        tensor([0., 1., 2., 3.])
        >>> freqt = diffsptk.FrequencyTransform(3, 4, 0.02)
        >>> c2 = freqt(c1)
        >>> c2
        tensor([ 0.0208,  1.0832,  2.1566,  2.9097, -0.1772])
        >>> freqt2 = diffsptk.FrequencyTransform(4, 3, -0.02)
        >>> c3 = freqt2(c2)
        >>> c3
        tensor([-9.8953e-10,  1.0000e+00,  2.0000e+00,  3.0000e+00])

        """
        check_size(c.size(-1), self.in_dim, "dimension of cepstrum")
        return self._forward(c, **self._buffers)

    @staticmethod
    def _func(c: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _, tensors = FrequencyTransform._precompute(
            c.size(-1) - 1, *args, **kwargs, device=c.device, dtype=c.dtype
        )
        return FrequencyTransform._forward(c, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(in_order: int, out_order: int, alpha: float) -> None:
        if in_order < 0:
            raise ValueError("in_order must be non-negative.")
        if out_order < 0:
            raise ValueError("out_order must be non-negative.")
        if 1 <= abs(alpha):
            raise ValueError("alpha must be in (-1, 1).")

    @staticmethod
    def _precompute(
        in_order: int,
        out_order: int,
        alpha: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        FrequencyTransform._check(in_order, out_order, alpha)
        L1 = in_order + 1
        L2 = out_order + 1
        beta = 1 - alpha * alpha

        ramp = torch.arange(L1, device=device, dtype=torch.double)
        A = torch.zeros((L2, L1), device=device, dtype=torch.double)
        A[0, :] = alpha**ramp
        if 1 < L2 and 1 < L1:
            A[1, 1:] = A[0, :-1] * beta * ramp[1:]
        for i in range(2, L2):
            i1 = i - 1
            for j in range(1, L1):
                j1 = j - 1
                A[i, j] = A[i1, j1] + alpha * (A[i, j1] - A[i1, j])

        return None, None, (to(A.T, dtype=dtype),)

    @staticmethod
    def _forward(c: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return torch.matmul(c, A)
