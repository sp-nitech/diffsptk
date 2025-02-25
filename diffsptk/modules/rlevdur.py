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

from ..misc.utils import check_size
from ..misc.utils import get_values
from ..misc.utils import remove_gain
from .base import BaseFunctionalModule


class ReverseLevinsonDurbin(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/rlevdur.html>`_
    for details.

    Parameters
    ----------
    lpc_order : int >= 0
        The order of the LPC coefficients, :math:`M`.

    """

    def __init__(self, lpc_order):
        super().__init__()

        self.in_dim = lpc_order + 1

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("eye", tensors[0])

    def forward(self, a):
        """Solve a Yule-Walker linear system given the LPC coefficients.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            The gain and the LPC coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The autocorrelation.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> acorr = diffsptk.Autocorrelation(5, 2)
        >>> levdur = diffsptk.LevinsonDurbin(2)
        >>> rlevdur = diffsptk.ReverseLevinsonDurbin(2)
        >>> r = acorr(x)
        >>> r
        tensor([ 5.8784,  0.8978, -2.0951])
        >>> r2 = rlevdur(levdur(r))
        >>> r2
        tensor([ 5.8784,  0.8978, -2.0951])

        """
        check_size(a.size(-1), self.in_dim, "dimension of LPC coefficients")
        return self._forward(a, **self._buffers)

    @staticmethod
    def _func(a, *args, **kwargs):
        _, _, tensors = ReverseLevinsonDurbin._precompute(
            a.size(-1) - 1, *args, **kwargs, device=a.device, dtype=a.dtype
        )
        return ReverseLevinsonDurbin._forward(a, *tensors)

    @staticmethod
    def _check(lpc_order):
        if lpc_order < 0:
            raise ValueError("lpc_order must be non-negative.")

    @staticmethod
    def _precompute(lpc_order, dtype=None, device=None):
        ReverseLevinsonDurbin._check(lpc_order)
        eye = torch.eye(lpc_order + 1, device=device, dtype=dtype)
        return None, None, (eye,)

    @staticmethod
    def _forward(a, eye):
        M = a.size(-1) - 1
        K, a = remove_gain(a, return_gain=True)

        U = [a.flip(-1)]
        E = [K**2]
        for m in range(M):
            u0 = U[-1][..., :1]
            u1 = U[-1][..., 1 : M - m]
            t = 1 / (1 - u0**2)
            u = (u1 - u0 * u1.flip(-1)) * t
            u = F.pad(u, (0, m + 2))
            e = E[-1] * t
            U.append(u)
            E.append(e)
        U = torch.stack(U[::-1], dim=-1)
        E = torch.stack(E[::-1], dim=-1)

        V = torch.linalg.solve_triangular(U, eye, upper=True, unitriangular=True)
        r = torch.matmul(V[..., :1].transpose(-2, -1) * E, V).squeeze(-2)
        return r
