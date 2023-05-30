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

import torch
import torch.nn as nn

from ..misc.utils import check_size
from ..misc.utils import is_power_of_two
from .vq import VectorQuantization


class LindeBuzoGrayAlgorithm(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lbg.html>`_
    for details. This module is not differentiable.

    order : int >= 0 [scalar]
        Order of vector.

    codebook_size : int >= 1 [scalar]
        Target codebook size, must be power of two.

    min_data_per_cluster : int >= 1 [scalar]
        Minimum number of data points in a cluster.

    n_iter : int >= 1 [scalar]
        Number of iterations.

    eps : float >= 0 [scalar]
        Convergence threshold.

    perturb_factor : float > 0 [scalar]
        Perturbation factor.

    verbose : bool [scalar]
        If True, print progress.

    """

    def __init__(
        self,
        order,
        codebook_size,
        min_data_per_cluster=1,
        n_iter=100,
        eps=1e-5,
        perturb_factor=1e-5,
        verbose=False,
    ):
        super(LindeBuzoGrayAlgorithm, self).__init__()

        self.order = order
        self.codebook_size = codebook_size
        self.min_data_per_cluster = min_data_per_cluster
        self.n_iter = n_iter
        self.eps = eps
        self.perturb_factor = perturb_factor
        self.verbose = verbose

        assert 0 <= self.order
        assert is_power_of_two(self.codebook_size)
        assert 1 <= self.min_data_per_cluster
        assert 1 <= self.n_iter
        assert 0 <= self.eps
        assert 0 < self.perturb_factor

        self.vq = VectorQuantization(order, codebook_size).eval()
        self.vq.codebook[:] = 1e10

        if self.verbose:
            self.logger = logging.getLogger("lbg")
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
            )
            self.logger.handlers.clear()
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def forward(self, x):
        """Design a codebook.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            Input vectors.

        Returns
        -------
        codebook : Tensor [shape=(K, M+1)]
            Codebook.

        indices : Tensor [shape=(...,)]
            Codebook indices.

        distance : Tensor [scalar]
            Distance.

        Examples
        --------
        >>> x = diffsptk.nrand(10, 0)
        >>> lbg = diffsptk.LBG(0, 2)
        >>> codebook, indices, distance = lbg(x)
        >>> codebook
        tensor([[-0.5277],
                [ 0.6747]])
        >>> indices
        tensor([0, 0, 0, 1, 0, 1, 1, 1, 1, 0])
        >>> distance
        tensor(0.2331)

        """
        check_size(x.size(-1), self.order + 1, "dimension of input")

        # Initalize codebook.
        x = x.view(-1, x.size(-1))
        mean = x.mean(0)
        self.vq.codebook[0] = mean
        distance = torch.inf

        curr_codebook_size = 1
        next_codebook_size = 2
        while next_codebook_size <= self.codebook_size:
            # Double codebook.
            codebook = self.vq.codebook[:curr_codebook_size]
            r = torch.randn_like(codebook) * self.perturb_factor
            self.vq.codebook[curr_codebook_size:next_codebook_size] = codebook - r
            self.vq.codebook[:curr_codebook_size] += r
            curr_codebook_size = next_codebook_size
            next_codebook_size *= 2
            if self.verbose:
                self.logger.info(f"K = {curr_codebook_size}")

            prev_distance = distance  # Suppress flake8 warnings.
            for n in range(self.n_iter):
                # E-step: evaluate model.
                xq, indices, _ = self.vq(x)
                distance = (x - xq).square().sum()
                distance /= x.size(0)
                if self.verbose:
                    self.logger.info(f"iter {n+1:5d}: distance = {distance:g}")

                # Check convergence.
                change = (prev_distance - distance).abs()
                if n and change / (distance + 1e-16) < self.eps:
                    break
                prev_distance = distance

                # Get number of data points for each cluster.
                n_data = torch.histc(
                    indices.float(),
                    bins=curr_codebook_size,
                    min=0,
                    max=curr_codebook_size - 1,
                )
                mask = self.min_data_per_cluster <= n_data

                # M-step: update centroids.
                centroids = torch.zeros(
                    (curr_codebook_size, self.order + 1), dtype=x.dtype, device=x.device
                )
                idx = indices.unsqueeze(1).expand(-1, self.order + 1)
                centroids.scatter_add_(0, idx, x)
                centroids[mask] /= n_data[mask].unsqueeze(1)

                if torch.any(~mask):
                    # Get index of largest cluster.
                    _, m = n_data.max(0)
                    copied_centroids = centroids[m : m + 1].expand((~mask).sum(), -1)
                    r = torch.randn_like(copied_centroids) * self.perturb_factor
                    centroids[~mask] = copied_centroids - r
                    centroids[m] += r.mean(0)

                self.vq.codebook[:curr_codebook_size] = centroids

        _, indices, _ = self.vq(x)

        return self.vq.codebook, indices, distance
