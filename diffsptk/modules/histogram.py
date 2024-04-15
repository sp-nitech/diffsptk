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

from ..misc.utils import to


class Histogram(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/histogram.html>`_
    for details.

    Parameters
    ----------
    n_bin : int >= 1
        Number of bins, :math:`K`.

    lower_bound : float < U
        Lower bound of the histogram, :math:`L`.

    upper_bound : float > L
        Upper bound of the histogram, :math:`U`.

    norm : bool
        If True, normalize the histogram.

    softness : float > 0
        A smoothing parameter. The smaller value makes the output closer to the true
        histogram, but the gradient vanishes.

    References
    ----------
    .. [1] M. Avi-Aharon et al., "DeepHist: Differentiable joint and color histogram
           layers for image-to-image translation," *arXiv preprint arXiv:2005.03995*,
           2020.

    """

    def __init__(
        self, n_bin=10, lower_bound=0, upper_bound=1, norm=False, softness=1e-3
    ):
        super().__init__()

        assert 1 <= n_bin
        assert lower_bound < upper_bound
        assert 0 < softness

        self.norm = norm
        self.softness = softness

        centers = self._precompute(n_bin, lower_bound, upper_bound)
        self.register_buffer("centers", centers)

    def forward(self, x):
        """Compute histogram.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Input data.

        Returns
        -------
        out : Tensor [shape=(..., K)]
            Histogram.

        Examples
        --------
        >>> x = diffsptk.ramp(9)
        >>> histogram = diffsptk.Histogram(n_bin=4, lower_bound=-0.1, upper_bound=9.1)
        >>> h = histogram(x)
        >>> h
        tensor([3., 2., 2., 3.])

        """
        return self._forward(x, self.norm, self.softness, self.centers)

    @staticmethod
    def _forward(x, norm, softness, centers):
        y = x.unsqueeze(-2) - centers.unsqueeze(-1)  # (..., K, T)
        g = 0.5 * (centers[1] - centers[0])
        h = torch.sigmoid((y + g) / softness) - torch.sigmoid((y - g) / softness)
        h = h.sum(-1)
        if norm:
            h /= h.sum(-1, keepdim=True)
        return h

    @staticmethod
    def _func(x, n_bin, lower_bound, upper_bound, norm, softness):
        centers = Histogram._precompute(
            n_bin, lower_bound, upper_bound, dtype=x.dtype, device=x.device
        )
        return Histogram._forward(x, norm, softness, centers)

    @staticmethod
    def _precompute(n_bin, lower_bound, upper_bound, dtype=None, device=None):
        width = (upper_bound - lower_bound) / n_bin
        bias = lower_bound + 0.5 * width
        centers = torch.arange(n_bin, dtype=torch.double, device=device) * width + bias
        return to(centers, dtype=dtype)
