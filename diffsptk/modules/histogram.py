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

from ..typing import Precomputed
from ..utils.private import filter_values, to
from .base import BaseFunctionalModule


class Histogram(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/histogram.html>`_
    for details. Note that the values at the edges of the bins cannot be counted
    correctly in the current implementation.

    Parameters
    ----------
    n_bin : int >= 1
        The number of bins, :math:`K`.

    lower_bound : float < U
        The lower bound of the histogram, :math:`L`.

    upper_bound : float > L
        The upper bound of the histogram, :math:`U`.

    norm : bool
        If True, normalizes the histogram.

    softness : float > 0
        A smoothing parameter. The smaller value makes the output closer to the true
        histogram, but the gradient vanishes.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] M. Avi-Aharon et al., "DeepHist: Differentiable joint and color histogram
           layers for image-to-image translation," *arXiv preprint arXiv:2005.03995*,
           2020.

    """

    def __init__(
        self,
        n_bin: int = 10,
        lower_bound: float = 0,
        upper_bound: float = 1,
        norm: bool = False,
        softness: float = 1e-3,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.values, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("centers", tensors[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute histogram.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The input data.

        Returns
        -------
        out : Tensor [shape=(..., K)]
            The histogram.

        Examples
        --------
        >>> x = diffsptk.ramp(9)
        >>> histogram = diffsptk.Histogram(n_bin=4, lower_bound=0, upper_bound=9)
        >>> h = histogram(x)
        >>> h
        tensor([2.5000, 2.0000, 2.0000, 2.5000])
        >>> histogram = diffsptk.Histogram(n_bin=4, lower_bound=-0.1, upper_bound=9.1)
        >>> h = histogram(x)
        >>> h
        tensor([3., 2., 2., 3.])

        """
        return self._forward(x, *self.values, **self._buffers)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, tensors = Histogram._precompute(
            *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return Histogram._forward(x, *values, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(
        n_bin: int, lower_bound: float, upper_bound: float, softness: float
    ) -> None:
        if n_bin <= 0:
            raise ValueError("n_bin must be positive.")
        if upper_bound <= lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound.")
        if softness <= 0:
            raise ValueError("softness must be positive.")

    @staticmethod
    def _precompute(
        n_bin: int,
        lower_bound: float,
        upper_bound: float,
        norm: bool,
        softness: float,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        Histogram._check(n_bin, lower_bound, upper_bound, softness)
        width = (upper_bound - lower_bound) / n_bin
        bias = lower_bound + 0.5 * width
        centers = torch.arange(n_bin, device=device, dtype=torch.double) * width + bias
        return (norm, softness), None, (to(centers, dtype=dtype),)

    @staticmethod
    def _forward(
        x: torch.Tensor, norm: bool, softness: float, centers: torch.Tensor
    ) -> torch.Tensor:
        y = x.unsqueeze(-2) - centers.unsqueeze(-1)  # (..., K, T)
        g = 0.5 * (centers[1] - centers[0])
        h = torch.sigmoid((y + g) / softness) - torch.sigmoid((y - g) / softness)
        h = h.sum(-1)
        if norm:
            h /= h.sum(-1, keepdim=True)
        return h
