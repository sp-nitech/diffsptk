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
import torch.nn.functional as F

from ..typing import Precomputed
from ..utils.private import filter_values
from .base import BaseFunctionalModule


class MedianFilter(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/medfilt.html>`_
    for details.

    Parameters
    ----------
    filter_length : int > 0
         The length of the median filter, :math:`L`.

    across_features : bool
        If True, apply the filter across the feature dimension.

    magic_number : float or None
        The magic number representing unvoiced frames.

    """

    def __init__(
        self,
        filter_length: int,
        across_features: bool = False,
        magic_number: float | None = None,
    ) -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply median filtering to the input sequence.

        Parameters
        ----------
        x : Tensor [shape=(B, N, D) or (N, D) or (N,)]
            The input sequence.

        Returns
        -------
        out : Tensor [shape=(B, N, D) or (B, N) or (N, D) or (N,)]
            The filtered sequence.

        Examples
        --------
        >>> import torch
        >>> import diffsptk
        >>> medfilt = diffsptk.MedianFilter(3)
        >>> x = torch.tensor([0, 2, -2, 7, 4, 8]).float()
        >>> y = medfilt(x)
        >>> y
        tensor([1., 0., 2., 4., 7., 6.])

        """
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = MedianFilter._precompute(*args, **kwargs)
        return MedianFilter._forward(x, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(filter_length: int) -> None:
        if filter_length <= 0:
            raise ValueError("filter_length must be positive.")

    @staticmethod
    def _precompute(
        filter_length: int, across_features: bool, magic_number: float | None
    ) -> Precomputed:
        MedianFilter._check(filter_length)
        if filter_length % 2 == 1:
            padding = ((filter_length - 1) // 2, (filter_length - 1) // 2)
        else:
            padding = (filter_length // 2, (filter_length - 2) // 2)
        padding = (0, 0) + padding  # No padding for feature dimension
        return (filter_length, padding, across_features, magic_number)

    @staticmethod
    def _forward(
        x: torch.Tensor,
        filter_length: int,
        padding: tuple[int, int],
        across_features: bool,
        magic_number: float | None,
    ) -> torch.Tensor:
        d = x.dim()
        if d == 1:
            x = x.reshape(1, -1, 1)
        elif d == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("Input must be 1D, 2D, or 3D tensor.")

        if magic_number is not None:
            mask = x == magic_number
            x = x.masked_fill(mask, float("nan"))

        y = F.pad(x, padding, value=float("nan"))
        y = y.unfold(1, filter_length, 1)
        if across_features:
            y = y.flatten(start_dim=-2)
        y = y.nanquantile(0.5, dim=-1)

        if magic_number is not None:
            m = F.pad(mask.float(), padding, value=float("nan"))
            m = m.unfold(1, filter_length, 1)
            if across_features:
                m = m.flatten(start_dim=-2)
            magic_count = m.nansum(dim=-1)
            valid_count = (1 - m).nansum(dim=-1)
            is_magic_dominant = magic_count > valid_count
            y = torch.where(is_magic_dominant, torch.full_like(y, magic_number), y)

        if d == 1:
            y = y.view(-1)
        elif d == 2:
            y = y.squeeze(0)
        return y
