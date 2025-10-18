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
from ..utils.private import filter_values, replicate1
from .base import BaseFunctionalModule


class ZeroCrossingAnalysis(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/zcross.html>`_
    for details.

    Parameters
    ----------
    frame_length : int >= 1
        The frame length in samples, :math:`L`.

    norm : bool
        If True, divide the zero-crossing rate by the frame length.

    softness : float > 0
        A smoothing parameter. The smaller value makes the output closer to the true
        zero-crossing rate, but the gradient vanishes.

    """

    def __init__(
        self, frame_length: int, norm: bool = False, softness: float = 1e-3
    ) -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute zero-crossing rate.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(..., T/L)]
            The zero-crossing rate.

        Examples
        --------
        >>> import diffsptk
        >>> import torch
        >>> zcross = diffsptk.ZeroCrossingAnalysis(3)
        >>> x = torch.tensor([-0.2, 0.3, -0.5, -0.7, 0.4, 0.2])
        >>> z = zcross(x)
        >>> z
        tensor([2., 1.])

        """
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = ZeroCrossingAnalysis._precompute(*args, **kwargs)
        return ZeroCrossingAnalysis._forward(x, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(frame_length: int, softness: float) -> None:
        if frame_length <= 0:
            raise ValueError("frame_length must be positive.")
        if softness <= 0:
            raise ValueError("softness must be positive.")

    @staticmethod
    def _precompute(frame_length: int, norm: bool, softness: float) -> Precomputed:
        ZeroCrossingAnalysis._check(frame_length, softness)
        return (frame_length, norm, softness)

    @staticmethod
    def _forward(
        x: torch.Tensor, frame_length: int, norm: bool, softness: float
    ) -> torch.Tensor:
        x = torch.tanh(x / softness)
        x = replicate1(x, right=False)
        x = x.unfold(-1, frame_length + 1, frame_length)
        z = 0.5 * torch.diff(x, dim=-1).abs().sum(-1)
        if norm:
            z /= frame_length
        return z
