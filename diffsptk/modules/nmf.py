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

from ..utils.private import get_generator, get_logger, to_dataloader
from .base import BaseLearnerModule


class NonnegativeMatrixFactorization(BaseLearnerModule):
    """Nonnegative matrix factorization module. Note that the forward method is not
    differentiable.

    Parameters
    ----------
    n_data : int >= 1
        The number of vectors, :math:`T`.

    order : int >= 0
        The order of the vector, :math:`M`.

    n_comp : int >= 1
        The number of principal components, :math:`K`.

    beta : float
        A control parameter of beta-divergence, :math:`\\beta`. 0: Itakura-Saito
        divergence, 1: generalized Kullback-Leibler divergence, 2: Euclidean distance.

    n_iter : int >= 1
        The number of iterations.

    eps : float >= 0
        The convergence threshold.

    act_norm : bool
        If True, normalizes the activation to sum to one.

    seed : int or None
        The random seed.

    batch_size : int >= 1 or None
        The bBatch size.

    verbose : bool or int
        If 1, shows the distance at each iteration; if 2, shows progress bars.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] M. Nakano et al., "Convergence-guaranteed multiplicative algorithms for
           nonnegative matrix factorization with beta-divergence," *IEEE International
           Workshop on Machine Learning for Signal Processing*, pp. 283-288, 2010.

    """

    def __init__(
        self,
        n_data: int,
        order: int,
        n_comp: int,
        *,
        beta: float = 0,
        n_iter: int = 100,
        eps: float = 1e-5,
        act_norm: bool = False,
        batch_size: int | None = None,
        seed: int | None = None,
        verbose: bool | int = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if n_data <= 0:
            raise ValueError("n_data must be positive.")
        if order < 0:
            raise ValueError("order must be non-negative.")
        if n_comp <= 0:
            raise ValueError("n_comp must be positive.")
        if n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if eps < 0:
            raise ValueError("eps must be non-negative.")

        self.beta = beta
        self.n_iter = n_iter
        self.eps = eps
        self.act_norm = act_norm
        self.batch_size = batch_size
        self.verbose = verbose

        generator = get_generator(seed, device=device)
        self.logger = get_logger("nmf")
        self.hide_progress_bar = self.verbose <= 1

        U = torch.rand(n_data, n_comp, device=device, dtype=dtype, generator=generator)
        if act_norm:
            U = U / U.sum(dim=1, keepdim=True)
        self.register_buffer("U", U)  # (T, K)

        H = torch.rand(
            n_comp, order + 1, device=device, dtype=dtype, generator=generator
        )
        self.register_buffer("H", H)  # (K, M+1)

        if beta < 1:
            phi = 1 / (2 - beta)
        elif 2 < beta:
            phi = 1 / (beta - 1)
        else:
            phi = 1
        self.phi = phi

    def warmup(
        self, x: torch.Tensor | torch.utils.data.DataLoader, **lbg_params
    ) -> None:
        """Initialize the dictionary matrix by K-means clustering.

        Parameters
        ----------
        x : Tensor [shape=(T, M+1)] or DataLoader
            The training data.

        lbg_params : additional keyword arguments
            The parameters for the Linde-Buzo-Gray algorithm.

        """
        x = to_dataloader(x, batch_size=self.batch_size)
        device = self.H.device

        from .lbg import LindeBuzoGrayAlgorithm

        K, L = self.H.shape
        lbg = LindeBuzoGrayAlgorithm(L - 1, K, **lbg_params).to(device)
        codebook, _ = lbg(x)
        self.H[:] = codebook

    @torch.inference_mode()
    def forward(
        self, x: torch.Tensor | torch.utils.data.DataLoader
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Estimate the coefficient matrix and dictionary matrix.

        Parameters
        ----------
        x : Tensor [shape=(T, M+1)] or DataLoader
            The input vectors or a DataLoader that yields the input vectors.

        Returns
        -------
        params : tuple of Tensors [shape=((T, K), (K, M+1))]
            The estimated coefficient matrix and dictionary matrix.

        divergence : Tensor [scalar]
            The divergence between the input and reconstructed vectors.

        Examples
        --------
        >>> x = diffsptk.nrand(10, 3) ** 2
        >>> nmf = diffsptk.NMF(10, 3, 2)
        >>> (U, H), _ = nmf(x)
        >>> U.shape
        torch.Size([10, 2])
        >>> H.shape
        torch.Size([2, 4])
        >>> y = U @ H
        >>> y.shape
        torch.Size([10, 4])

        """
        x = to_dataloader(x, self.batch_size)
        device = self.H.device

        prev_divergence = torch.inf
        for n in range(self.n_iter):
            # Update parameters.
            H_numer = 0
            H_denom = 0
            t1 = 0
            for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
                t2 = t1 + len(batch_x)

                # Check input vectors.
                if n == 0:
                    if batch_x.dim() != 2:
                        raise ValueError("Input vectors must be 2D.")
                    if (batch_x <= 0).any():
                        raise ValueError("Input vectors must be positive.")

                # Update coefficient matrix.
                z = batch_x.to(device)
                y = torch.matmul(self.U[t1:t2], self.H)  # (T, M+1)
                y2 = z * y ** (self.beta - 2)
                y1 = y ** (self.beta - 1)
                U_numer = torch.matmul(y2, self.H.T)  # (T, K)
                U_denom = torch.matmul(y1, self.H.T)  # (T, K)
                self.U[t1:t2] *= (U_numer / U_denom) ** self.phi
                if self.act_norm:
                    self.U[t1:t2] /= self.U[t1:t2].sum(dim=1, keepdim=True)

                # Accumulate statistics using the updated coefficient matrix.
                y = torch.matmul(self.U[t1:t2], self.H)  # (T, M+1)
                y2 = z * y ** (self.beta - 2)
                y1 = y ** (self.beta - 1)
                H_numer += torch.matmul(self.U[t1:t2].T, y2)  # (K, M+1)
                H_denom += torch.matmul(self.U[t1:t2].T, y1)  # (K, M+1)

                t1 = t2

            # Update dictionary matrix.
            self.H *= (H_numer / H_denom) ** self.phi

            # Calculate divergence.
            divergence = 0
            t1 = 0
            for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
                t2 = t1 + len(batch_x)
                z = batch_x.to(device)
                y = torch.matmul(self.U[t1:t2], self.H)
                if self.beta == 0:
                    term1 = z / y
                    term2 = torch.log(term1)
                    divergence += (term1 - term2 - 1).sum()
                elif self.beta == 1:
                    term1 = z * torch.log(z / y)
                    term2 = z - y
                    divergence += (term1 - term2).sum()
                else:
                    beta1 = self.beta - 1
                    term1 = z * (z**beta1 - y**beta1) / beta1
                    term2 = (z**self.beta - y**self.beta) / self.beta
                    divergence += (term1 - term2).sum()
                t1 = t2

            if self.verbose:
                self.logger.info(f"  iter {n + 1:5d}: divergence = {divergence:g}")

            # Check convergence.
            change = (prev_divergence - divergence).abs()
            if n and change / (divergence + 1e-16) < self.eps:
                break
            prev_divergence = divergence

        return (self.U, self.H), divergence

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the input vectors using the estimated parameters.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            The input vectors.

        Returns
        -------
        out : Tensor [shape=(..., K)]
            The transformed vectors.

        """
        return torch.matmul(x, self.H.T)
