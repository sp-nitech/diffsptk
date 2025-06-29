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
from .pca import PrincipalComponentAnalysis


class IndependentComponentAnalysis(BaseLearnerModule):
    """Independent component analysis module. Note that the forward method is not
    differentiable.

    Parameters
    ----------
    order : int >= 0
        The order of the vector, :math:`M`.

    n_comp : int >= 1
        The number of components, :math:`K`.

    func : ['logcosh', 'gauss']
        The nonquadratic function used in the approximation of negentropy.

    n_iter : int >= 1
        The number of iterations.

    eps : float >= 0
        The convergence threshold.

    batch_size : int >= 1 or None
        The batch size.

    seed : int or None
        The random seed.

    verbose : bool
        If True, shows progress bars.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] A. Hyvarinen and E. Oja, "Independent component analysis: algorithms and
           applications," *Neural Networks*, vol. 13, pp. 411-430, 2000.

    """

    def __init__(
        self,
        order: int,
        n_comp: int,
        *,
        func: str = "logcosh",
        n_iter: int = 100,
        eps: float = 1e-4,
        batch_size: int | None = None,
        seed: int | None = None,
        verbose: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if order <= 0:
            raise ValueError("order must be positive.")
        if order + 1 < n_comp:
            raise ValueError("n_comp must be less than or equal to input dimension.")
        if n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if eps < 0:
            raise ValueError("eps must be non-negative.")

        self.n_iter = n_iter
        self.eps = eps
        self.batch_size = batch_size
        self.verbose = verbose

        generator = get_generator(seed, device=device)
        self.logger = get_logger("ica")
        self.hide_progress_bar = self.verbose <= 1

        if func == "logcosh":
            self.g = torch.tanh
            self.g_prime = lambda u: 1 - torch.tanh(u) ** 2
        elif func == "gauss":
            self.g = lambda u: u * torch.exp(-(u**2) / 2)
            self.g_prime = lambda u: (1 - u**2) * torch.exp(-(u**2) / 2)
        else:
            raise ValueError(f"func {func} is not supported.")

        self.pca = PrincipalComponentAnalysis(
            order, n_comp, batch_size=batch_size, device=device, dtype=dtype
        )

        W = torch.randn(n_comp, n_comp, device=device, dtype=dtype, generator=generator)
        self.register_buffer("W", W)  # (K, K)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform independent component analysis.

        Parameters
        ----------
        x : Tensor [shape=(T, M+1)] or DataLoader
            The input vectors or a DataLoader that yields the input vectors.

        Returns
        -------
        W : Tensor [shape=(K, K)]
            The separating matrix.

        """
        x = to_dataloader(x, self.batch_size)
        device = self.W.device

        # Obtain whitening matrix.
        self.pca(x)

        def decorrelate(W):
            s, V = torch.linalg.eigh(torch.matmul(W, W.T))
            d = 1 / torch.sqrt(torch.clip(s, min=1e-10))
            W = torch.matmul(torch.matmul(V * d, V.T), W)
            return W

        W = decorrelate(self.W)

        for n in range(self.n_iter):
            prev_W = W

            # Update separating matrix.
            T = term1 = term2 = 0
            for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
                xp = batch_x.to(device)
                xq = self.pca.whiten(self.pca.center(xp))  # (T, K)
                Wx = torch.matmul(W, xq.T)  # (K, T)
                T += len(xq)
                term1 += torch.matmul(self.g(Wx), xq)  # (K, K)
                term2 += W * self.g_prime(Wx).sum(dim=1, keepdim=True)  # (K, K)
            W = (term1 - term2) / T
            W = decorrelate(W)

            # Check convergence.
            similarity = torch.diagonal(torch.matmul(W, prev_W.T)).abs()
            criterion = (similarity - 1).abs().max()
            if self.verbose:
                self.logger.info(f"  iter {n + 1:5d}: criterion = {criterion:g}")
            if criterion < self.eps:
                break

        # Scale separating matrix as we cannot determine the scale.
        s2 = 0
        for (batch_x,) in tqdm(x, disable=self.hide_progress_bar):
            xp = batch_x.to(device)
            s = self.transform(xp)
            s2 += s.square().sum(0)
        W /= torch.sqrt(s2 / T).unsqueeze(-1)

        self.W[:] = W
        return self.W

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Separate the input vectors into independent components.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            The input vectors.

        Returns
        -------
        out : Tensor [shape=(..., K)]
            The estimated independent components.

        """
        return torch.matmul(self.pca.whiten(self.pca.center(x)), self.W.T)
