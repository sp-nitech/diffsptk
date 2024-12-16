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
from tqdm import tqdm

from ..misc.utils import outer
from ..misc.utils import to_dataloader


class PrincipalComponentAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/pca.html>`_
    for details. Note that the forward method is not differentiable.

    Parameters
    ----------
    order : int >= 0
        Order of vector.

    n_comp : int >= 1
        Number of principal components, :math:`K`.

    cov_type : ['sample', 'unbiased', 'correlation']
        Type of covariance.

    sort : ['ascending', 'descending']
        Order of eigenvalues and eigenvectors.

    batch_size : int >= 1 or None
        Batch size.

    verbose : bool
        If True, show progress bar.

    """

    def __init__(
        self,
        order,
        n_comp,
        *,
        cov_type="sample",
        sort="descending",
        batch_size=None,
        verbose=False,
    ):
        super().__init__()

        assert 1 <= n_comp <= order + 1
        assert sort in ["ascending", "descending"]

        self.n_comp = n_comp
        self.sort = sort
        self.batch_size = batch_size
        self.hide_progress_bar = not verbose

        def sample_cov(x0, x1, x2):
            return x2 / x0 - torch.outer(x1, x1) / (x0 * x0)

        if cov_type in (0, "sample"):

            def cov(x0, x1, x2):
                return sample_cov(x0, x1, x2)
        elif cov_type in (1, "unbiased"):

            def cov(x0, x1, x2):
                c = sample_cov(x0, x1, x2)
                return c * (x0 / (x0 - 1))
        elif cov_type in (2, "correlation"):

            def cov(x0, x1, x2):
                c = sample_cov(x0, x1, x2)
                v = c.diag().sqrt()
                return c / torch.outer(v, v)
        else:
            raise ValueError(f"cov_type {cov_type} is not supported.")
        self.cov = cov

        self.register_buffer("s", torch.zeros(n_comp))
        self.register_buffer("V", torch.eye(n_comp, order + 1))
        self.register_buffer("m", torch.zeros(order + 1))

    def forward(self, x):
        """Perform PCA.

        Parameters
        ----------
        x : Tensor [shape=(T, M+1)] or DataLoader
            Input vectors or dataloader yielding input vectors.

        Returns
        -------
        s : Tensor [shape=(K,)]
            Eigenvalues.

        V : Tensor [shape=(K, M+1)]
            Eigenvectors.

        m : Tensor [shape=(M+1,)]
            Mean vector.

        Examples
        --------
        >>> x = diffsptk.nrand(10, 3)
        >>> x.size()
        torch.Size([10, 4])
        >>> pca = diffsptk.PCA(3, 3)
        >>> s, _, _ = pca(x)
        >>> s
        tensor([1.3465, 0.7497, 0.4447])
        >>> y = pca.transform(x)
        >>> y.size()
        torch.Size([10, 3])

        """
        x = to_dataloader(x, self.batch_size)
        device = self.m.device

        # Compute statistics.
        x0 = x1 = x2 = 0
        for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
            assert batch_x.dim() == 2
            xp = batch_x.to(device)
            x0 += xp.size(0)
            x1 += xp.sum(0)
            x2 += outer(xp).sum(0)

        if x0 <= self.n_comp:
            raise RuntimeError("Number of data samples is too small.")

        # Compute mean and covariance matrix.
        m = x1 / x0
        c = self.cov(x0, x1, x2)

        # Compute eigenvalues and eigenvectors.
        val, vec = torch.linalg.eigh(c)
        val = val[-self.n_comp :]
        vec = vec[:, -self.n_comp :]
        if self.sort == "descending":
            val = val.flip(-1)
            vec = vec.flip(-1)
        self.s[:] = val
        self.V[:] = vec.T
        self.m[:] = m
        return self.s, self.V, self.m

    def transform(self, x):
        """Transform input vectors using estimated eigenvectors.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            Input vectors.

        Returns
        -------
        out : Tensor [shape=(..., K)]
            Transformed vectors.

        """
        V = self.V.T.flip(-1) if self.sort == "ascending" else self.V.T
        return torch.matmul(self.center(x), V)

    def center(self, x):
        """Center input vectors using estimated mean.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            Input vectors.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            Centered vectors.

        """
        return x - self.m

    def whiten(self, x):
        """Whiten input vectors using estimated parameters.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            Input vectors.

        Returns
        -------
        out : Tensor [shape=(..., K)]
            Whitened vectors.

        """
        V = self.V.T.flip(-1) if self.sort == "ascending" else self.V.T
        s = self.s.flip(-1) if self.sort == "ascending" else self.s
        d = torch.sqrt(torch.clip(s, min=1e-10))
        return torch.matmul(x, V / d)
