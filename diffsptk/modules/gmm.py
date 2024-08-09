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

import logging

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ..misc.utils import to_dataloader
from .lbg import LindeBuzoGrayAlgorithm


class GaussianMixtureModeling(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/gmm.html>`_
    for details. This module is not differentiable.

    Parameters
    ----------
    order : int >= 0
        Order of vector, :math:`M`.

    n_mixture : int >= 1
        Number of mixture components, :math:`K`.

    n_iter : int >= 1
        Number of iterations.

    eps : float >= 0
        Convergence threshold.

    weight_floor : float >= 0
        Floor value for mixture weights.

    var_floor : float >= 0
        Floor value for variance.

    var_type : ['diag', 'full']
        Type of covariance.

    block_size : list[int]
        Block size of covariance matrix.

    ubm : tuple of Tensors [shape=((K,), (K, M+1), (K, M+1, M+1))]
        Parameters of universal background model.

    alpha : float in [0, 1]
        Smoothing parameter.

    batch_size : int >= 1 or None
        Batch size.

    verbose : bool
        If True, print progress.

    """

    def __init__(
        self,
        order,
        n_mixture,
        n_iter=100,
        eps=1e-5,
        weight_floor=1e-5,
        var_floor=1e-6,
        var_type="diag",
        block_size=None,
        ubm=None,
        alpha=0,
        batch_size=None,
        verbose=False,
    ):
        super().__init__()

        assert 0 <= order
        assert 1 <= n_mixture
        assert 1 <= n_iter
        assert 0 <= eps
        assert 0 <= weight_floor <= 1 / n_mixture
        assert 0 <= var_floor
        assert 0 <= alpha <= 1

        self.order = order
        self.n_mixture = n_mixture
        self.n_iter = n_iter
        self.eps = eps
        self.weight_floor = weight_floor
        self.var_floor = var_floor
        self.alpha = alpha
        self.batch_size = batch_size
        self.verbose = verbose

        if self.alpha != 0:
            assert ubm is not None

        # Check block size.
        L = self.order + 1
        if block_size is None:
            block_size = [L]
        assert sum(block_size) == L

        self.is_diag = var_type == "diag" and len(block_size) == 1

        # Make mask for covariance matrix.
        mask = torch.zeros((L, L))
        cumsum = np.cumsum(np.insert(block_size, 0, 0))
        for b1, s1, e1 in zip(block_size, cumsum[:-1], cumsum[1:]):
            assert 0 < b1
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
        K = self.n_mixture
        self.register_buffer("w", torch.ones(K) / K)
        self.register_buffer("mu", torch.randn(K, L))
        self.register_buffer("sigma", torch.eye(L).repeat(K, 1, 1))

        # Save UBM parameters.
        if ubm is not None:
            self.set_params(ubm)
            ubm_w, ubm_mu, ubm_sigma = ubm
            self.register_buffer("ubm_w", ubm_w)
            self.register_buffer("ubm_mu", ubm_mu)
            self.register_buffer("ubm_sigma", ubm_sigma)

        if self.verbose:
            self.logger = logging.getLogger("gmm")
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
            )
            self.logger.handlers.clear()
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def set_params(self, params):
        """Set model parameters.

        Parameters
        ----------
        params : tuple of Tensors [shape=((K,), (K, M+1), (K, M+1, M+1))]
            Parameters of Gaussian mixture model.

        """
        w, mu, sigma = params
        if w is not None:
            self.w[:] = w
        if mu is not None:
            self.mu[:] = mu
        if sigma is not None:
            self.sigma[:] = sigma

    def warmup(self, x, **lbg_params):
        """Initialize model parameters by K-means clustering.

        Parameters
        ----------
        x : Tensor [shape=(T, M+1)] or DataLoader
            Training data.

        lbg_params : additional keyword arguments
            Parameters for Linde-Buzo-Gray algorithm.

        Returns
        -------
        out : tuple of Tensors [shape=((K,), (K, M+1), (K, M+1, M+1))]
            GMM parameters.

        """
        x = to_dataloader(x, batch_size=self.batch_size)
        device = self.w.device

        lbg = LindeBuzoGrayAlgorithm(self.order, self.n_mixture, **lbg_params).to(
            device
        )
        codebook, indices, _ = lbg(x, return_indices=True)

        count = torch.bincount(indices, minlength=self.n_mixture).to(self.w.dtype)
        w = count / len(indices)
        mu = codebook

        idx = indices.view(-1, 1, 1).expand(-1, self.order + 1, self.order + 1)
        kxx = torch.zeros_like(self.sigma)  # [K, L, L]
        b = 0
        for (batch_x,) in tqdm(x, disable=self.verbose <= 1):
            e = b + batch_x.size(0)
            xp = batch_x.to(device)
            xx = torch.matmul(xp.unsqueeze(-1), xp.unsqueeze(-2))
            kxx.scatter_add_(0, idx[b:e], xx)
            b = e
        mm = torch.matmul(mu.unsqueeze(-1), mu.unsqueeze(-2))  # [K, L, L]
        sigma = kxx / count.view(-1, 1, 1) - mm
        sigma = sigma * self.mask

        params = (w, mu, sigma)
        self.set_params(params)
        return params

    def forward(self, x, return_posterior=False):
        """Train Gaussian mixture models.

        Parameters
        ----------
        x : Tensor [shape=(T, M+1)] or DataLoader
            Input vectors or dataloder yielding input vectors.

        return_posterior : bool
            If True, return posterior probabilities.

        Returns
        -------
        params : tuple of Tensors [shape=((K,), (K, M+1), (K, M+1, M+1))]
            GMM parameters.

        posterior : Tensor [shape=(T, K)] (optional)
            Posterior probabilities.

        log_likelihood : Tensor [scalar]
            Total log-likelihood.

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
            def e_step():
                log_pi = (self.order + 1) * np.log(2 * np.pi)
                if self.is_diag:
                    log_det = torch.log(
                        torch.diagonal(self.sigma, dim1=-2, dim2=-1)
                    ).sum(-1)  # [K]
                    precision = torch.reciprocal(
                        torch.diagonal(self.sigma, dim1=-2, dim2=-1)
                    )  # [K, L]
                    mahala = []
                    for (batch_x,) in tqdm(x, disable=self.verbose <= 1):
                        xp = batch_x.to(device)
                        diff = xp.unsqueeze(1) - self.mu.unsqueeze(0)  # [B, K, L]
                        mahala.append((diff**2 * precision).sum(-1))  # [B, K]
                    mahala = torch.cat(mahala)  # [T, K]
                else:
                    col = torch.linalg.cholesky(self.sigma)
                    log_det = (
                        torch.log(torch.diagonal(col, dim1=-2, dim2=-1)).sum(-1) * 2
                    )  # [K]
                    precision = torch.cholesky_inverse(col).unsqueeze(0)  # [1, K, L, L]
                    mahala = []
                    for (batch_x,) in tqdm(x, disable=self.verbose <= 1):
                        xp = batch_x.to(device)
                        diff = xp.unsqueeze(1) - self.mu.unsqueeze(0)  # [B, K, L]
                        right = torch.matmul(
                            precision, diff.unsqueeze(-1)
                        )  # [B, K, L, 1]
                        mahala.append(
                            torch.matmul(diff.unsqueeze(-2), right)
                            .squeeze(-1)
                            .squeeze(-1)
                        )  # [B, K]
                    mahala = torch.cat(mahala)  # [T, K]
                numer = torch.log(self.w) - 0.5 * (log_pi + log_det + mahala)  # [T, K]
                denom = torch.logsumexp(numer, dim=-1, keepdim=True)  # [T, 1]
                posterior = torch.exp(numer - denom)  # [T, K]
                log_likelihood = torch.sum(denom)
                return posterior, log_likelihood

            posterior, log_likelihood = e_step()

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
            self.w = torch.clamp(self.w, min=self.weight_floor)
            sum_floor = self.weight_floor * self.n_mixture
            a = (1 - sum_floor) / (self.w.sum() - sum_floor)
            b = self.weight_floor * (1 - a)
            self.w = a * self.w + b

            # Update mean vectors.
            px = []
            b = 0
            for (batch_x,) in tqdm(x, disable=self.verbose <= 1):
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
                for (batch_x,) in tqdm(x, disable=self.verbose <= 1):
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
                    nn = nu**2
                    a = pxx - y.view(-1, 1) * (2 * nn - mm)
                    b = xi.view(-1, 1) * self.ubm_sigma.diagonal(dim1=-2, dim2=-1)
                    diff = self.ubm_mu - self.mu
                    dd = diff**2
                    c = xi.view(-1, 1) * dd
                    sigma = (a + b + c) * z.view(-1, 1)
                self.sigma.diagonal(dim1=-2, dim2=-1).copy_(sigma)
            else:
                pxx = []
                b = 0
                for (batch_x,) in tqdm(x, disable=self.verbose <= 1):
                    e = b + batch_x.size(0)
                    xp = batch_x.to(device)
                    xx = torch.matmul(xp.unsqueeze(-1), xp.unsqueeze(-2))
                    pxx.append(torch.einsum("bk,blm->klm", posterior[b:e], xx))
                    b = e
                pxx = sum(pxx)
                mm = torch.matmul(self.mu.unsqueeze(-1), self.mu.unsqueeze(-2))
                if self.alpha == 0:
                    sigma = pxx * z.view(-1, 1, 1) - mm
                else:
                    y = posterior.sum(dim=0)
                    nu = px / y.view(-1, 1)
                    nm = torch.matmul(nu.unsqueeze(-1), self.mu.unsqueeze(-2))
                    mn = nm.transpose(-2, -1)
                    a = pxx - y.view(-1, 1, 1) * (nm + mn - mm)
                    b = xi.view(-1, 1, 1) * self.ubm_sigma
                    diff = self.ubm_mu - self.mu
                    dd = torch.matmul(diff.unsqueeze(-1), diff.unsqueeze(-2))
                    c = xi.view(-1, 1, 1) * dd
                    sigma = (a + b + c) * z.view(-1, 1, 1)
                self.sigma = sigma * self.mask
            self.sigma.diagonal(dim1=-2, dim2=-1).clamp_(min=self.var_floor)

            # Check convergence.
            change = log_likelihood - prev_log_likelihood
            if self.verbose:
                self.logger.info(f"iter {n+1:5d}: average = {log_likelihood / T:g}")
            if n and change < self.eps:
                break
            prev_log_likelihood = log_likelihood

        ret = [[self.w, self.mu, self.sigma]]

        if return_posterior:
            posterior, _ = e_step()
            ret.append(posterior)

        ret.append(log_likelihood)
        return ret
