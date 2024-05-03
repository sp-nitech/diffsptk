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


class PrincipalComponentAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/pca.html>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        Order of vector.

    n_comp : int >= 1
        Number of principal components, :math:`N`.

    cov_type : ['sample', 'unbiased', 'correlation']
        Type of covariance.

    sort : ['ascending', 'descending']
        Order of eigenvalues and eigenvectors.

    """

    def __init__(self, order, n_comp, cov_type="sample", sort="descending"):
        super().__init__()

        assert 0 <= order
        assert 1 <= n_comp <= order + 1
        assert sort in ["ascending", "descending"]

        self.order = order
        self.n_comp = n_comp
        self.cov_type = cov_type
        self.sort = sort

        if cov_type in (0, "sample"):
            self.cov = lambda x: torch.cov(x, correction=0)
        elif cov_type in (1, "unbiased"):
            self.cov = lambda x: torch.cov(x, correction=1)
        elif cov_type in (2, "correlation"):
            self.cov = lambda x: torch.corrcoef(x)
        else:
            raise ValueError(f"cov_type {cov_type} is not supported.")

        self.register_buffer("v", torch.eye(self.order + 1, self.n_comp))
        self.register_buffer("m", torch.zeros(self.order + 1))

    def forward(self, x):
        """Perform PCA.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            Input vectors.

        Returns
        -------
        e : Tensor [shape=(N,)]
            Eigenvalues.

        v : Tensor [shape=(M+1, N)]
            Eigenvectors.

        m : Tensor [shape=(M+1,)]
            Mean vector.

        Examples
        --------
        >>> x = diffsptk.nrand(10, 3)
        >>> x.size()
        torch.Size([10, 4])
        >>> pca = diffsptk.PCA(3, 3)
        >>> e, _, _ = pca(x)
        >>> e
        tensor([1.3465, 0.7497, 0.4447])
        >>> y = pca.transform(x)
        >>> y.size()
        torch.Size([10, 3])

        """
        check_size(x.size(-1), self.order + 1, "dimension of input")

        x = x.reshape(-1, x.size(-1)).T
        assert self.n_comp + 1 <= x.size(1), "Number of data samples is too small"

        e, v = torch.linalg.eigh(self.cov(x))
        e = e[-self.n_comp :]
        v = v[:, -self.n_comp :]
        if self.sort == "descending":
            e = e.flip(-1)
            v = v.flip(-1)
        self.v[:] = v
        self.m[:] = x.mean(1)
        return e, self.v, self.m

    def transform(self, x):
        """Transform input vectors using estimated eigenvectors.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            Input vectors.

        Returns
        -------
        out : Tensor [shape=(..., N)]
            Transformed vectors.

        """
        v = self.v.flip(-1) if self.sort == "ascending" else self.v
        return torch.matmul(x - self.m, v)
