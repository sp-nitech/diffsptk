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

from typing import TypeAlias

import numpy as np
import torch
from tqdm import tqdm

from ..typing import ArrayLike
from ..utils.private import get_logger, outer, to, to_dataloader
from .base import BaseLearnerModule

Params: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class GaussianMixtureModeling(BaseLearnerModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/gmm.html>`_
    for details. Note that the forward method is not differentiable.

    Parameters
    ----------
    order : int >= 0
        The order of the vector, :math:`M`.

    n_mixture : int >= 1
        The number of mixture components, :math:`K`.

    n_iter : int >= 1
        The number of iterations.

    eps : float >= 0
        The convergence threshold.

    weight_floor : float >= 0
        The floor value for mixture weights.

    var_floor : float >= 0
        The floor value for variance.

    var_type : ['diag', 'full']
        The type of covariance matrix.

    block_size : list[int]
        The block size of covariance matrix.

    ubm : tuple of Tensors [shape=((K,), (K, M+1), (K, M+1, M+1))]
        The GMM parameters of a universal background model.

    alpha : float in [0, 1]
        The smoothing parameter.

    batch_size : int >= 1 or None
        The batch size.

    verbose : bool
        If 1, shows the likelihood at each iteration; if 2, shows progress bars.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] J-L. Gauvain et al., "Maximum a posteriori estimation for multivariate
           Gaussian mixture observations of Markov chains," *IEEE Transactions on Speech
           and Audio Processing*, vol. 2, no. 2, pp. 291-298, 1994.

    """

    def __init__(
        self,
        order: int,
        n_mixture: int,
        *,
        n_iter: int = 100,
        eps: float = 1e-5,
        weight_floor: float = 1e-5,
        var_floor: float = 1e-6,
        var_type: str = "diag",
        block_size: ArrayLike[int] | None = None,
        ubm: Params | None = None,
        alpha: float = 0,
        batch_size: int | None = None,
        verbose: bool | int = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if order < 0:
            raise ValueError("order must be non-negative.")
        if n_mixture <= 0:
            raise ValueError("n_mixture must be positive.")
        if n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if eps < 0:
            raise ValueError("eps must be non-negative.")
        if not 0 <= weight_floor <= 1 / n_mixture:
            raise ValueError("weight_floor must be in [0, 1 / K].")
        if var_floor < 0:
            raise ValueError("var_floor must be non-negative.")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1].")

        self.order = order
        self.n_mixture = n_mixture
        self.n_iter = n_iter
        self.eps = eps
        self.weight_floor = weight_floor
        self.var_floor = var_floor
        self.alpha = alpha
        self.batch_size = batch_size
        self.verbose = verbose

        self.logger = get_logger("gmm")
        self.hide_progress_bar = self.verbose <= 1

        if self.alpha != 0 and ubm is None:
            raise ValueError("ubm must be provided when alpha is not 0.")

        # Check block size.
        L = self.order + 1
        if block_size is None:
            block_size = [L]
        if sum(block_size) != L:
            raise ValueError("The sum of block_size must be equal to order + 1.")
        if not all(0 < b for b in block_size):
            raise ValueError("All elements of block_size must be positive.")

        self.is_diag = var_type == "diag" and len(block_size) == 1

        # Make mask for covariance matrix.
        mask = torch.zeros((L, L), device=device, dtype=dtype)
        cumsum = np.cumsum(np.insert(block_size, 0, 0))
        for b1, s1, e1 in zip(block_size, cumsum[:-1], cumsum[1:]):
            if var_type == "diag":
                for b2, s2, e2 in zip(block_size, cumsum[:-1], cumsum[1:]):
                    if b1 == b2:
                        mask[s1:e1, s2:e2] = torch.eye(b1)
            elif var_type == "full":
                mask[s1:e1, s1:e1] = 1
            else:
                raise ValueError(f"var_type {var_type} is not supported.")
        self.register_buffer("mask", mask)

        # Initialize model parameters.
        params = {"device": device, "dtype": dtype}
        K = self.n_mixture
        self.register_buffer("w", torch.ones(K, **params) / K)
        self.register_buffer("mu", torch.randn(K, L, **params))
        self.register_buffer("sigma", torch.eye(L, **params).repeat(K, 1, 1))

        # Save UBM parameters.
        if ubm is not None:
            self.set_params(ubm)
            ubm_w, ubm_mu, ubm_sigma = ubm
            self.register_buffer("ubm_w", to(ubm_w, **params))
            self.register_buffer("ubm_mu", to(ubm_mu, **params))
            self.register_buffer("ubm_sigma", to(ubm_sigma, **params))

    def set_params(
        self,
        params: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    ) -> None:
        """Set model parameters.

        Parameters
        ----------
        params : tuple of Tensors [shape=((K,), (K, M+1), (K, M+1, M+1))]
            The GMM parameters.

        """
        w, mu, sigma = params
        if w is not None:
            self.w[:] = w
        if mu is not None:
            self.mu[:] = mu
        if sigma is not None:
            self.sigma[:] = sigma

    def warmup(
        self, x: torch.Tensor | torch.utils.data.DataLoader, **lbg_params
    ) -> None:
        """Initialize the model parameters by K-means clustering.

        Parameters
        ----------
        x : Tensor [shape=(T, M+1)] or DataLoader
            The training data.

        lbg_params : additional keyword arguments
            The parameters for the Linde-Buzo-Gray algorithm.

        """
        x = to_dataloader(x, batch_size=self.batch_size)
        device = self.w.device

        from .lbg import LindeBuzoGrayAlgorithm

        lbg = LindeBuzoGrayAlgorithm(self.order, self.n_mixture, **lbg_params).to(
            device
        )
        codebook, indices, _ = lbg(x, return_indices=True)

        count = torch.bincount(indices, minlength=self.n_mixture).to(self.w.dtype)
        w = count / len(indices)
        mu = codebook

        idx = indices.view(-1, 1, 1).expand(-1, self.order + 1, self.order + 1)
        kxx = torch.zeros_like(self.sigma)  # (K, L, L)
        b = 0
        for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
            e = b + batch_x.size(0)
            xx = outer(batch_x.to(device))
            kxx.scatter_add_(0, idx[b:e], xx)
            b = e
        mm = outer(mu)  # (K, L, L)
        sigma = kxx / count.view(-1, 1, 1) - mm
        sigma = sigma * self.mask

        params = (w, mu, sigma)
        self.set_params(params)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor | torch.utils.data.DataLoader,
        return_posterior: bool = False,
    ) -> tuple[Params, torch.Tensor] | tuple[Params, torch.Tensor, torch.Tensor]:
        """Train Gaussian mixture models.

        Parameters
        ----------
        x : Tensor [shape=(T, M+1)] or DataLoader
            The input vectors or a DataLoader that yields the input vectors.

        return_posterior : bool
            If True, return the posterior probabilities.

        Returns
        -------
        params : tuple of Tensors [shape=((K,), (K, M+1), (K, M+1, M+1))]
            The estimated GMM parameters.

        posterior : Tensor [shape=(T, K)] (optional)
            The posterior probabilities.

        log_likelihood : Tensor [scalar]
            The total log-likelihood.

        Examples
        --------
        >>> x = diffsptk.nrand(10, 1)
        >>> gmm = diffsptk.GMM(1, 2)
        >>> params, log_likelihood = gmm(x)
        >>> w, mu, sigma = params
        >>> w
        tensor([0.1917, 0.8083])
        >>> mu
        tensor([[ 1.2321,  0.2058],
                [-0.1326, -0.7006]])
        >>> sigma
        tensor([[[3.4010e-01, 0.0000e+00],
                 [0.0000e+00, 6.2351e-04]],
                [[3.0944e-01, 0.0000e+00],
                 [0.0000e+00, 8.6096e-01]]])
        >>> log_likelihood
        tensor(-19.5235)

        """
        x = to_dataloader(x, batch_size=self.batch_size)
        device = self.w.device

        prev_log_likelihood = -torch.inf
        for n in range(self.n_iter):
            # Compute log probabilities.
            posterior, log_likelihood = self._e_step(x)

            # Update mixture weights.
            T = len(posterior)
            if self.alpha == 0:
                z = posterior.sum(dim=0)
                self.w = z / T
            else:
                xi = self.ubm_w * self.alpha
                z = posterior.sum(dim=0) + xi
                self.w = z / (T + self.alpha)
            z = 1 / z
            self.w = torch.clip(self.w, min=self.weight_floor)
            sum_floor = self.weight_floor * self.n_mixture
            a = (1 - sum_floor) / (self.w.sum() - sum_floor)
            b = self.weight_floor * (1 - a)
            self.w = a * self.w + b

            # Update mean vectors.
            px = []
            b = 0
            for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
                e = b + batch_x.size(0)
                px.append(torch.matmul(posterior[b:e].t(), batch_x.to(device)))
                b = e
            px = sum(px)
            if self.alpha == 0:
                self.mu = px * z.view(-1, 1)
            else:
                self.mu = (px + xi.view(-1, 1) * self.ubm_mu) * z.view(-1, 1)

            # Update covariance matrices.
            if self.is_diag:
                pxx = []
                b = 0
                for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
                    e = b + batch_x.size(0)
                    xx = batch_x.to(device) ** 2
                    pxx.append(torch.matmul(posterior[b:e].t(), xx))
                    b = e
                pxx = sum(pxx)
                mm = self.mu**2
                if self.alpha == 0:
                    sigma = pxx * z.view(-1, 1) - mm
                else:
                    y = posterior.sum(dim=0)
                    nu = px / y.view(-1, 1)
                    nm = nu * self.mu
                    a = pxx - y.view(-1, 1) * (2 * nm - mm)
                    b = xi.view(-1, 1) * self.ubm_sigma.diagonal(dim1=-2, dim2=-1)
                    c = xi.view(-1, 1) * (self.ubm_mu - self.mu) ** 2
                    sigma = (a + b + c) * z.view(-1, 1)
                self.sigma.diagonal(dim1=-2, dim2=-1).copy_(sigma)
            else:
                pxx = []
                b = 0
                for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
                    e = b + batch_x.size(0)
                    xx = outer(batch_x.to(device))
                    pxx.append(torch.einsum("bk,blm->klm", posterior[b:e], xx))
                    b = e
                pxx = sum(pxx)
                mm = outer(self.mu)
                if self.alpha == 0:
                    sigma = pxx * z.view(-1, 1, 1) - mm
                else:
                    y = posterior.sum(dim=0)
                    nu = px / y.view(-1, 1)
                    nm = outer(nu, self.mu)
                    mn = nm.transpose(-2, -1)
                    a = pxx - y.view(-1, 1, 1) * (nm + mn - mm)
                    b = xi.view(-1, 1, 1) * self.ubm_sigma
                    c = xi.view(-1, 1, 1) * outer(self.ubm_mu - self.mu)
                    sigma = (a + b + c) * z.view(-1, 1, 1)
                self.sigma = sigma * self.mask
            self.sigma.diagonal(dim1=-2, dim2=-1).clip_(min=self.var_floor)

            # Check convergence.
            if self.verbose:
                self.logger.info(f"iter {n + 1:5d}: average = {log_likelihood / T:g}")
            change = log_likelihood - prev_log_likelihood
            if n and change < self.eps:
                break
            prev_log_likelihood = log_likelihood

        params = (self.w, self.mu, self.sigma)
        if return_posterior:
            posterior, _ = self._e_step(x)
            return params, posterior, log_likelihood
        return params, log_likelihood

    def transform(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Transform the input vectors based on a single mixture sequence.

        Parameters
        ----------
        x : Tensor [shape=(T, N+1)]
            The input vectors.

        Returns
        -------
        y : Tensor [shape=(T, M-N)]
            The output vectors.

        indices : Tensor [shape=(T,)]
            The selected mixture indices.

        log_prob : Tensor [shape=(T,)]
            The log probabilities.

        """
        N = x.size(-1) - 1
        posterior, log_prob = self._e_step(x, reduction="none", in_order=N)
        indices = torch.argmax(posterior, dim=-1)

        if self.order == N:
            return None, indices, log_prob

        L = N + 1
        sigma_yx = self.sigma[:, L:, :L]
        sigma_xx = self.sigma[:, :L, :L]
        sigma_yx_xx = torch.matmul(sigma_yx, torch.inverse(sigma_xx))
        mu_x = self.mu[indices, :L]
        mu_y = self.mu[indices, L:]
        diff = (x - mu_x).unsqueeze(-1)
        E = mu_y + torch.matmul(sigma_yx_xx[indices], diff).squeeze(-1)
        y = E
        return y, indices, log_prob

    def _e_step(
        self,
        x: torch.Tensor | torch.utils.data.DataLoader,
        reduction: str = "sum",
        in_order: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = to_dataloader(x, self.batch_size)
        device = self.w.device

        if in_order is None:
            L = self.order + 1
            mu, sigma = self.mu, self.sigma
        else:
            L = in_order + 1
            mu, sigma = self.mu[:, :L], self.sigma[:, :L, :L]

        log_pi = L * np.log(2 * np.pi)
        if self.is_diag:
            log_det = torch.log(torch.diagonal(sigma, dim1=-2, dim2=-1)).sum(-1)  # (K,)
            precision = torch.reciprocal(
                torch.diagonal(sigma, dim1=-2, dim2=-1)
            )  # (K, L)
            mahala = []
            for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
                xp = batch_x.to(device)
                diff = xp.unsqueeze(1) - mu.unsqueeze(0)  # (B, K, L)
                mahala.append((diff**2 * precision).sum(-1))  # (B, K)
            mahala = torch.cat(mahala)  # (T, K)
        else:
            col = torch.linalg.cholesky(sigma)
            log_det = (
                torch.log(torch.diagonal(col, dim1=-2, dim2=-1)).sum(-1) * 2
            )  # (K,)
            precision = torch.cholesky_inverse(col).unsqueeze(0)  # (1, K, L, L)
            mahala = []
            for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
                xp = batch_x.to(device)
                diff = xp.unsqueeze(1) - mu.unsqueeze(0)  # (B, K, L)
                right = torch.matmul(precision, diff.unsqueeze(-1))  # (B, K, L, 1)
                mahala.append(
                    torch.matmul(diff.unsqueeze(-2), right).squeeze(-1).squeeze(-1)
                )  # (B, K)
            mahala = torch.cat(mahala)  # (T, K)
        numer = torch.log(self.w) - 0.5 * (log_pi + log_det + mahala)  # (T, K)
        denom = torch.logsumexp(numer, dim=-1, keepdim=True)  # (T, 1)
        posterior = torch.exp(numer - denom)  # (T, K)
        if reduction == "none":
            log_likelihood = denom.squeeze(-1)
        elif reduction == "sum":
            log_likelihood = torch.sum(denom)
        else:
            raise ValueError(f"reduction {reduction} is not supported.")
        return posterior, log_likelihood
