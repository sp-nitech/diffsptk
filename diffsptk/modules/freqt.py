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

from ..misc.utils import check_size
from ..misc.utils import to


class FrequencyTransform(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/freqt.html>`_
    for details.

    Parameters
    ----------
    in_order : int >= 0
        Order of input sequence, :math:`M_1`.

    out_order : int >= 0
        Order of output sequence, :math:`M_2`.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    """

    def __init__(self, in_order, out_order, alpha=0):
        super().__init__()

        assert 0 <= in_order
        assert 0 <= out_order
        assert abs(alpha) < 1

        self.in_order = in_order
        self.out_order = out_order
        self.register_buffer(
            "A", self._precompute(self.in_order, self.out_order, alpha)
        )

    def forward(self, c):
        """Perform frequency transform.

        Parameters
        ----------
        c : Tensor [shape=(..., M1+1)]
            Input sequence.

        Returns
        -------
        out : Tensor [shape=(..., M2+1)]
            Warped sequence.

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
        check_size(c.size(-1), self.in_order + 1, "dimension of cepstrum")
        return self._forward(c, self.A)

    @staticmethod
    def _forward(c, A):
        return torch.matmul(c, A)

    @staticmethod
    def _func(c, out_order, alpha):
        in_order = c.size(-1) - 1
        A = FrequencyTransform._precompute(
            in_order, out_order, alpha, dtype=c.dtype, device=c.device
        )
        return FrequencyTransform._forward(c, A)

    @staticmethod
    def _precompute(in_order, out_order, alpha, dtype=None, device=None):
        L1 = in_order + 1
        L2 = out_order + 1
        beta = 1 - alpha * alpha

        # Make transform matrix.
        arange = torch.arange(L1, dtype=torch.double, device=device)
        A = torch.zeros((L2, L1), dtype=torch.double, device=device)
        A[0, :] = alpha**arange
        if 1 < L2 and 1 < L1:
            A[1, 1:] = A[0, :-1] * beta * arange[1:]
        for i in range(2, L2):
            i1 = i - 1
            for j in range(1, L1):
                j1 = j - 1
                A[i, j] = A[i1, j1] + alpha * (A[i, j1] - A[i1, j])
        return to(A.T, dtype=dtype)
