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
from tqdm import tqdm

from ..utils.private import outer, to_dataloader
from .base import BaseLearnerModule


class PrincipalComponentAnalysis(BaseLearnerModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/pca.html>`_
    for details. Note that the forward method is not differentiable.

    Parameters
    ----------
    order : int >= 0
        The order of the vector, :math:`M`.

    n_comp : int >= 1
        The number of principal components, :math:`K`.

    cov_type : ['sample', 'unbiased', 'correlation']
        The type of covariance matrix.

    sort : ['ascending', 'descending']
        The order of eigenvalues.

    batch_size : int >= 1 or None
        The batch size.

    verbose : bool
        If True, shows progress bars.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        order: int,
        n_comp: int,
        *,
        cov_type: str | int = "sample",
        sort: str = "descending",
        batch_size: int | None = None,
        verbose: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if order < 0:
            raise ValueError("order must be non-negative.")
        if order + 1 < n_comp:
            raise ValueError("n_comp must be less than or equal to input dimension.")
        if sort not in ["ascending", "descending"]:
            raise ValueError("sort must be 'ascending' or 'descending'.")

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

        params = {"device": device, "dtype": dtype}
        self.register_buffer("s", torch.zeros(n_comp, **params))
        self.register_buffer("V", torch.eye(n_comp, order + 1, **params))
        self.register_buffer("m", torch.zeros(order + 1, **params))

    def forward(
        self, x: torch.Tensor | torch.utils.data.DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform principal component analysis.

        Parameters
        ----------
        x : Tensor [shape=(T, M+1)] or DataLoader
            The input vectors or a DataLoader that yields the input vectors.

        Returns
        -------
        s : Tensor [shape=(K,)]
            The eigenvalues.

        V : Tensor [shape=(K, M+1)]
            The eigenvectors.

        m : Tensor [shape=(M+1,)]
            The mean vector.

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
            if batch_x.dim() != 2:
                raise ValueError("Input vectors must be 2D.")
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

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the input vectors using the estimated eigenvectors.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            The input vectors.

        Returns
        -------
        out : Tensor [shape=(..., K)]
            The transformed vectors.

        """
        V = self.V.T.flip(-1) if self.sort == "ascending" else self.V.T
        return torch.matmul(self.center(x), V)

    def center(self, x: torch.Tensor) -> torch.Tensor:
        """Center the input vectors using the estimated mean.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            The input vectors.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The centered vectors.

        """
        return x - self.m

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        """Whiten the input vectors using the estimated parameters.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            The input vectors.

        Returns
        -------
        out : Tensor [shape=(..., K)]
            The whitened vectors.

        """
        V = self.V.T.flip(-1) if self.sort == "ascending" else self.V.T
        s = self.s.flip(-1) if self.sort == "ascending" else self.s
        d = torch.sqrt(torch.clip(s, min=1e-10))
        return torch.matmul(x, V / d)
