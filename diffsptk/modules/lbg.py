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

import math

import torch
from torch import nn
from tqdm import tqdm

from ..misc.utils import get_generator
from ..misc.utils import get_logger
from ..misc.utils import to_dataloader
from .gmm import GaussianMixtureModeling
from .vq import VectorQuantization


class LindeBuzoGrayAlgorithm(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lbg.html>`_
    for details. Note that the forward method is not differentiable.

    Parameters
    ----------
    order : int >= 0
        Order of vector, :math:`M`.

    codebook_size : int >= 1
        Target codebook size, :math:`K`.

    min_data_per_cluster : int >= 1
        Minimum number of data points in a cluster.

    n_iter : int >= 1
        Number of iterations.

    eps : float >= 0
        Convergence threshold.

    perturb_factor : float > 0
        Perturbation factor.

    init : ['none', 'mean'] or torch.Tensor [shape=(1~K, M+1)]
        Initialization type.

    metric : ['none, 'aic', 'bic']
        Metric used as a reference for model selection.

    batch_size : int >= 1 or None
        Batch size.

    seed : int or None
        Random seed.

    verbose : bool or int
        If 1, show distance at each iteration; if 2, show progress bar.

    References
    ----------
    .. [1] Y. Linde et al., "An algorithm for vector quantizer design," *IEEE
           Transactions on Communications*, vol. 28, no. 1, pp. 84-95, 1980.

    """

    def __init__(
        self,
        order,
        codebook_size,
        *,
        min_data_per_cluster=1,
        n_iter=100,
        eps=1e-5,
        perturb_factor=1e-5,
        init="mean",
        metric="none",
        batch_size=None,
        seed=None,
        verbose=False,
    ):
        super().__init__()

        assert 0 <= order
        assert 1 <= codebook_size
        assert 1 <= min_data_per_cluster
        assert 1 <= n_iter
        assert 0 <= eps
        assert 0 < perturb_factor

        self.order = order
        self.codebook_size = codebook_size
        self.min_data_per_cluster = min_data_per_cluster
        self.n_iter = n_iter
        self.eps = eps
        self.perturb_factor = perturb_factor
        self.metric = metric
        self.batch_size = batch_size
        self.verbose = verbose

        self.generator = get_generator(seed)
        self.logger = get_logger("lbg")
        self.hide_progress_bar = self.verbose <= 1

        self.vq = VectorQuantization(order, codebook_size).eval()

        if torch.is_tensor(init):
            given_codebook_size = init.size(0)
            c = codebook_size
            while c % 2 == 0 and c != given_codebook_size:
                c //= 2
            if c != given_codebook_size:
                raise ValueError(
                    "Codebook size must be a power-of-two muptiple of "
                    "the initial codebook size."
                )
            self.curr_codebook_size = given_codebook_size
            self.init = "none"
            self.vq.codebook[:given_codebook_size] = init
        else:
            c = codebook_size
            while c % 2 == 0:
                c //= 2
            self.curr_codebook_size = c
            self.init = init

    def forward(self, x, return_indices=False):
        """Design a codebook.

        Parameters
        ----------
        x : Tensor [shape=(T, M+1)] or DataLoader
            Input vectors or dataloder yielding input vectors.

        return_indices : bool
            If True, return indices.

        Returns
        -------
        codebook : Tensor [shape=(K, M+1)]
            Codebook.

        indices : Tensor [shape=(T,)] (optional)
            Codebook indices.

        distance : Tensor [scalar]
            Distance.

        Examples
        --------
        >>> x = diffsptk.nrand(10, 0)
        >>> lbg = diffsptk.LBG(0, 2)
        >>> codebook, indices, distance = lbg(x, return_indices=True)
        >>> codebook
        tensor([[-0.5277],
                [ 0.6747]])
        >>> indices
        tensor([0, 0, 0, 1, 0, 1, 1, 1, 1, 0])
        >>> distance
        tensor(0.2331)

        """
        x = to_dataloader(x, self.batch_size)
        device = self.vq.codebook.device

        # Initialize codebook.
        if self.init == "none":
            pass
        elif self.init == "mean":
            if self.verbose:
                self.logger.info("K = 1")
            s = T = 0
            for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
                assert batch_x.dim() == 2
                batch_xp = batch_x.to(device)
                s += batch_xp.sum(0)
                T += batch_xp.size(0)
            self.vq.codebook[0] = s / T
        else:
            raise ValueError(f"init {self.init} is not supported.")
        self.vq.codebook[self.curr_codebook_size :] = 1e10

        def e_step(x):
            indices = []
            distance = 0
            for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
                batch_xp = batch_x.to(device)
                batch_xq, batch_indices, _ = self.vq(batch_xp)
                indices.append(batch_indices)
                distance += (batch_xp - batch_xq).square().sum()
            indices = torch.cat(indices)
            distance /= len(indices)
            return indices, distance

        distance = torch.inf
        while True:
            next_codebook_size = self.curr_codebook_size * 2
            if next_codebook_size <= self.codebook_size:
                # Double codebook.
                codebook = self.vq.codebook[: self.curr_codebook_size]
                r = torch.randn_like(codebook) * self.perturb_factor
                self.vq.codebook[self.curr_codebook_size : next_codebook_size] = (
                    codebook - r
                )
                self.vq.codebook[: self.curr_codebook_size] += r
                self.curr_codebook_size = next_codebook_size
                next_codebook_size *= 2

            if self.verbose:
                self.logger.info(f"K = {self.curr_codebook_size}")

            prev_distance = distance  # Suppress flake8 warnings.
            for n in range(self.n_iter):
                # E-step: evaluate model.
                indices, distance = e_step(x)
                if self.verbose:
                    self.logger.info(f"  iter {n + 1:5d}: distance = {distance:g}")

                # Check convergence.
                change = (prev_distance - distance).abs()
                if n and change / (distance + 1e-16) < self.eps:
                    break
                prev_distance = distance

                # Get number of data points for each cluster.
                n_data = torch.histc(
                    indices.float(),
                    bins=self.curr_codebook_size,
                    min=0,
                    max=self.curr_codebook_size - 1,
                )
                mask = self.min_data_per_cluster <= n_data

                # M-step: update centroids.
                centroids = torch.zeros(
                    (self.curr_codebook_size, self.order + 1),
                    dtype=distance.dtype,
                    device=device,
                )
                idx = indices.unsqueeze(1).expand(-1, self.order + 1)
                b = 0
                for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
                    e = b + batch_x.size(0)
                    centroids.scatter_add_(0, idx[b:e], batch_x.to(device))
                    b = e
                centroids[mask] /= n_data[mask].unsqueeze(1)

                if torch.any(~mask):
                    # Get index of largest cluster.
                    m = torch.argmax(n_data, 0)
                    copied_centroids = centroids[m : m + 1].expand((~mask).sum(), -1)
                    r = (
                        torch.randn(
                            copied_centroids.size(),
                            dtype=copied_centroids.dtype,
                            device=device,
                            generator=self.generator,
                        )
                        * self.perturb_factor
                    )
                    centroids[~mask] = copied_centroids - r
                    centroids[m] += r.mean(0)

                self.vq.codebook[: self.curr_codebook_size] = centroids

            if self.metric != "none":
                gmm = GaussianMixtureModeling(self.order, self.curr_codebook_size)
                gmm.set_params((None, centroids, None))
                _, log_likelihood = gmm._e_step(x)
                n_param = self.curr_codebook_size * (self.order + 1)
                if self.metric == "aic":
                    metric = -2 * log_likelihood + n_param * 2
                elif self.metric == "bic":
                    metric = -2 * log_likelihood + n_param * math.log(len(indices))
                else:
                    raise ValueError(f"metric {self.metric} is not supported.")
                if self.verbose:
                    self.logger.info(f"  {self.metric.upper()} = {metric:g}")

            if self.curr_codebook_size == self.codebook_size:
                break

        ret = [self.vq.codebook]

        if return_indices:
            indices, _ = e_step(x)
            ret.append(indices)

        ret.append(distance)
        return ret

    def transform(self, x):
        """Transform input vectors using the codebook.

        Parameters
        ----------
        x : Tensor [shape=(T, M+1)]
            Input vectors.

        Returns
        -------
        xq : Tensor [shape=(T, M+1)]
            Quantized vectors.

        indices : Tensor [shape=(T,)]
            Codebook indices.

        Examples
        --------
        >>> lbg = diffsptk.LBG(0, 2)
        >>> torch.save(lbg.state_dict(), "lbg.pt")
        >>> lbg.load_state_dict(torch.load("lbg.pt"))
        >>> x = diffsptk.nrand(10, 0)
        >>> xq, indices = lbg.transform(x)

        """
        xq, indices, _ = self.vq(x)
        return xq, indices
