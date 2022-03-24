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

import numpy as np
import torch
import torch.nn as nn

from ..misc.utils import default_dtype


class FrequencyTransform(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/freqt.html>`_
    for details.

    Parameters
    ----------
    in_order : int >= 0 [scalar]
        Order of input sequence, :math:`M_1`.

    out_order : int >= 0 [scalar]
        Order of output sequence, :math:`M_2`.

    alpha : float [-1 < alpha < 1]
        Frequency warping factor, :math:`\\alpha`.

    """

    def __init__(self, in_order, out_order, alpha):
        super(FrequencyTransform, self).__init__()

        assert 0 <= in_order
        assert 0 <= out_order
        assert abs(alpha) < 1

        beta = 1 - alpha * alpha
        L1 = in_order + 1
        L2 = out_order + 1

        # Make transform matrix.
        A = np.zeros((L2, L1), dtype=default_dtype())
        A[0, :] = alpha ** np.arange(L1)
        if 1 < L2 and 1 < L1:
            A[1, 1:] = alpha ** np.arange(L1 - 1) * np.arange(1, L1) * beta
        for i in range(2, L2):
            i1 = i - 1
            for j in range(1, L1):
                j1 = j - 1
                A[i, j] = A[i1, j1] + alpha * (A[i, j1] - A[i1, j])

        self.register_buffer("A", torch.from_numpy(A).t())

    def forward(self, x):
        """Perform frequency transform.

        Parameters
        ----------
        x : Tensor [shape=(..., M1+1)]
            Input sequence.

        Returns
        -------
        y : Tensor [shape=(..., M2+1)]
            Warped sequence.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        tensor([0., 1., 2., 3.])
        >>> freqt = diffsptk.FrequencyTransform(3, 4, 0.02)
        >>> y = freqt(x)
        >>> y
        tensor([ 0.0208,  1.0832,  2.1566,  2.9097, -0.1772])
        >>> freqt2 = diffsptk.FrequencyTransform(4, 3, -0.02)
        >>> x2 = freqt2(y)
        >>> x2
        tensor([-9.8953e-10,  1.0000e+00,  2.0000e+00,  3.0000e+00])

        """
        y = torch.matmul(x, self.A)
        return y
