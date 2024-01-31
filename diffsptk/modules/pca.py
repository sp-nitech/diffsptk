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


class PrincipalComponentAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/pca.html>`_
    for details.

    Parameters
    ----------
    n_comp : int >= 1 [scalar]
        Number of principal components, :math:`N`.

    cov_type : ['sample', 'unbiased', 'correlation']
        Type of covariance.

    """

    def __init__(self, n_comp, cov_type="sample"):
        super(PrincipalComponentAnalysis, self).__init__()

        self.n_comp = n_comp
        self.cov_type = cov_type

        assert 1 <= self.n_comp

        if cov_type == 0 or cov_type == "sample":
            self.cov = lambda x: torch.cov(x, correction=0)
        elif cov_type == 1 or cov_type == "unbiased":
            self.cov = lambda x: torch.cov(x, correction=1)
        elif cov_type == 2 or cov_type == "correlation":
            self.cov = lambda x: torch.corrcoef(x)
        else:
            raise ValueError(f"cov_type {cov_type} is not supported")

    def forward(self, x):
        """Perform PCA.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            Input vectors.

        Returns
        -------
        e : Tensor [shape=(N,)]
            Eigenvalues ordered in ascending order.

        v : Tensor [shape=(M+1, N)]
            Eigenvectors.

        m : Tensor [shape=(M+1,)]
            Mean vector.

        Examples
        --------
        >>> x = diffsptk.nrand(10, 3)
        >>> pca = diffsptk.PCA(3)
        >>> e, _, _ = pca(x)
        >>> e
        tensor([0.6240, 1.0342, 1.7350])

        """
        x = x.reshape(-1, x.size(-1)).T
        assert self.n_comp + 1 <= x.size(1), "Number of data samples is too small"

        e, v = torch.linalg.eigh(self.cov(x))
        e = e[-self.n_comp :]
        v = v[:, -self.n_comp :]
        m = x.mean(1)
        return e, v, m
