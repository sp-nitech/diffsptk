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
import torch.nn.functional as F

from ..misc.utils import to


class StructuralSimilarityIndex(nn.Module):
    """Structural similarity index computation.

    Parameters
    ----------
    reduction : ['none', 'mean', 'sum']
        Reduction type.

    alpha : float > 0
        Relative importance of luminance component.

    beta : float > 0
        Relative importance of contrast component.

    gamma : float > 0
        Relative importance of structure component.

    kernel_size : int >= 1
        Kernel size of Gaussian filter.

    sigma : float > 0
        Standard deviation of Gaussian filter.

    k1 : float > 0
        A small constant.

    k2 : float > 0
        A small constant.

    eps : float >= 0
        A small value to prevent NaN.

    padding : ['valid', 'same']
        Padding type.

    dynamic_range : float > 0 or None
        Dynamic range of input. If None, input is automatically normalized.

    References
    ----------
    .. [1] Z. Wang et al., "Image quality assessment: From error visibility to
           structural similarity," *IEEE Transactions on Image Processing*, vol. 13,
           no. 4, pp. 600-612, 2004.

    """

    def __init__(
        self,
        reduction="mean",
        *,
        alpha=1,
        beta=1,
        gamma=1,
        kernel_size=11,
        sigma=1.5,
        k1=0.01,
        k2=0.03,
        eps=1e-8,
        padding="same",
        dynamic_range=None,
    ):
        super().__init__()

        assert reduction in ["none", "mean", "sum"]
        assert 1 <= kernel_size and kernel_size % 2 == 1
        assert 0 < sigma
        assert 0 < k1 < 1
        assert 0 < k2 < 1
        assert 0 <= eps

        self.reduction = reduction
        self.weights = (alpha, beta, gamma)
        self.ks = (k1, k2)
        self.eps = eps
        self.padding = padding
        self.dynamic_range = dynamic_range
        self.register_buffer("kernel", self._precompute(kernel_size, sigma))

    def forward(self, x, y):
        """Calculate SSIM.

        Parameters
        ----------
        x : Tensor [shape=(..., N, D)]
            Input.

        y : Tensor [shape=(..., N, D)]
            Target.

        Returns
        -------
        out : Tensor [shape=(..., N, D) or scalar]
            SSIM or mean SSIM.

        Examples
        --------
        >>> x = diffsptk.nrand(20, 20)
        >>> y = diffsptk.nrand(20, 20)
        >>> ssim = diffsptk.StructuralSimilarityIndex()
        >>> s = ssim(x, y)
        >>> s
        tensor(0.0588)

        """
        return self._forward(
            x,
            y,
            self.reduction,
            self.weights,
            self.ks,
            self.eps,
            self.padding,
            self.dynamic_range,
            self.kernel,
        )

    @staticmethod
    def _forward(x, y, reduction, weights, ks, eps, padding, dynamic_range, kernel):
        org_shape = x.shape
        x = x.view(-1, 1, x.shape[-2], x.shape[-1])
        y = y.view(-1, 1, y.shape[-2], y.shape[-1])

        # Normalize x and y to [0, 1].
        if dynamic_range is None:
            x_max = torch.amax(x, dim=(-2, -1), keepdim=True)
            x_min = torch.amin(x, dim=(-2, -1), keepdim=True)
            y_max = torch.amax(y, dim=(-2, -1), keepdim=True)
            y_min = torch.amin(y, dim=(-2, -1), keepdim=True)
            xy_max = torch.maximum(x_max, y_max)
            xy_min = torch.minimum(x_min, y_min)
            d = xy_max - xy_min + eps
            x = (x - xy_min) / d
            y = (y - xy_min) / d
            dynamic_range = 1

        # Pad x and y.
        if padding == "valid":
            pass
        elif padding == "same":
            pad_size = kernel.shape[-1] // 2
            x = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode="reflect")
            y = F.pad(y, (pad_size, pad_size, pad_size, pad_size), mode="reflect")
        else:
            raise ValueError(f"padding {padding} is not supported.")

        # Set constants.
        K1, K2 = ks
        L = dynamic_range
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        C3 = 0.5 * C2

        # Calculate luminance.
        mu_x = F.conv2d(x, kernel, padding=0)
        mu_y = F.conv2d(y, kernel, padding=0)
        mu_x2 = mu_x**2
        mu_y2 = mu_y**2
        luminance = (2 * mu_x * mu_y + C1) / (mu_x2 + mu_y2 + C1)

        # Calculate contrast.
        sigma_x2 = torch.clip(F.conv2d(x**2, kernel, padding=0) - mu_x2, min=eps)
        sigma_y2 = torch.clip(F.conv2d(y**2, kernel, padding=0) - mu_y2, min=eps)
        sigma_x = torch.sqrt(sigma_x2)
        sigma_y = torch.sqrt(sigma_y2)
        contrast = (2 * sigma_x * sigma_y + C2) / (sigma_x2 + sigma_y2 + C2)

        # Calculate structure.
        mu_xy = mu_x * mu_y
        sigma_xy = F.conv2d(x * y, kernel, padding=0) - mu_xy
        structure = (sigma_xy + C3) / (sigma_x * sigma_y + C3)

        # Calculate SSIM.
        alpha, beta, gamma = weights
        ssim = (luminance**alpha) * (contrast**beta) * (structure**gamma)
        ssim = ssim.view(*org_shape[:-2], *ssim.shape[-2:])

        if reduction == "none":
            pass
        elif reduction == "sum":
            ssim = ssim.sum()
        elif reduction == "mean":
            ssim = ssim.mean()
        else:
            raise ValueError(f"reduction {reduction} is not supported.")
        return ssim

    @staticmethod
    def _func(
        x,
        y,
        reduction,
        alpha,
        beta,
        gamma,
        kernel_size,
        sigma,
        k1,
        k2,
        eps,
        padding,
        dynamic_range,
    ):
        kernel = StructuralSimilarityIndex._precompute(
            kernel_size, sigma, dtype=x.dtype, device=x.device
        )
        return StructuralSimilarityIndex._forward(
            x,
            y,
            reduction,
            (alpha, beta, gamma),
            (k1, k2),
            eps,
            padding,
            dynamic_range,
            kernel,
        )

    @staticmethod
    def _precompute(kernel_size, sigma, dtype=None, device=None):
        # Generate 2D Gaussian kernel.
        center = kernel_size // 2
        x = torch.arange(kernel_size, dtype=torch.double, device=device) - center
        xx = x**2
        G = torch.exp(-0.5 * (xx.unsqueeze(0) + xx.unsqueeze(1)) / sigma**2)
        G /= G.sum()  # Normalized to unit sum.
        G = G.view(1, 1, kernel_size, kernel_size)
        return to(G, dtype=dtype)
