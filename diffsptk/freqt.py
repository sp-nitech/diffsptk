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


class FrequencyTransform(nn.Module):
    def __init__(self, in_order, out_order, alpha):
        """Initialize module.

        Parameters
        ----------
        in_order : int >= 0 [scalar]
            Order of input sequence, m.

        out_order : int >= 0 [scalar]
            Order of output sequence, M.

        alpha : float [-1 < alpha < 1]
            Frequency warping factor.

        """
        super(FrequencyTransform, self).__init__()

        self.in_order = in_order
        self.out_order = out_order
        self.alpha = alpha

        assert 0 <= self.in_order
        assert 0 <= self.out_order
        assert abs(self.alpha) < 1

        beta = 1 - self.alpha * self.alpha
        L1 = self.in_order + 1
        L2 = self.out_order + 1

        # Make transform matrix.
        A = np.zeros((L2, L1), dtype=np.float32)
        A[0, :] = self.alpha ** np.arange(L1)
        if 1 < L2 and 1 < L1:
            A[1, 1:] = self.alpha ** np.arange(L1 - 1) * np.arange(1, L1) * beta
        for i in range(2, L2):
            i1 = i - 1
            for j in range(1, L1):
                j1 = j - 1
                A[i, j] = A[i1, j1] + self.alpha * (A[i, j1] - A[i1, j])

        self.register_buffer("A", torch.from_numpy(A).t())

    def forward(self, x):
        """Perform frequency transform.

        Parameters
        ----------
        x : Tensor [shape=(..., m+1)]
            Input sequence.

        Returns
        -------
        y : Tensor [shape=(..., M+1)]
            Warped sequence.

        """
        y = torch.matmul(x, self.A if x.dtype == self.A.dtype else self.A.double())
        return y
