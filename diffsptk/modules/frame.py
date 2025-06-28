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


class Frame(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/frame.html>`_
    for details.

    Parameters
    ----------
    frame_length : int >= 1
        The frame length in samples, :math:`L`.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    center : bool
        If True, pad the input on both sides so that the frame is centered.

    zmean : bool
        If True, perform mean subtraction on each frame.

    mode : ['constant', 'reflect', 'replicate', 'circular']
        The padding method.

    """

    def __init__(
        self,
        frame_length: int,
        frame_period: int,
        *,
        center: bool = True,
        zmean: bool = False,
        mode: str = "constant",
    ) -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply framing to the given waveform.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The waveform.

        Returns
        -------
        out : Tensor [shape=(..., T/P, L)]
            The framed waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 9)
        >>> frame = diffsptk.Frame(5, 2)
        >>> y = frame(x)
        >>> y
        tensor([[0., 0., 1., 2., 3.],
                [1., 2., 3., 4., 5.],
                [3., 4., 5., 6., 7.],
                [5., 6., 7., 8., 9.],
                [7., 8., 9., 0., 0.]])

        """
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = Frame._precompute(*args, **kwargs)
        return Frame._forward(x, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(frame_length: int, frame_period: int) -> None:
        if frame_length <= 0:
            raise ValueError("frame_length must be positive.")
        if frame_period <= 0:
            raise ValueError("frame_period must be positive.")

    @staticmethod
    def _precompute(
        frame_length: int,
        frame_period: int,
        center: bool = True,
        zmean: bool = False,
        mode: str = "constant",
    ) -> Precomputed:
        Frame._check(frame_length, frame_period)
        return (
            frame_length,
            frame_period,
            center,
            zmean,
            mode,
        )

    @staticmethod
    def _forward(
        x: torch.Tensor,
        frame_length: int,
        frame_period: int,
        center: bool,
        zmean: bool,
        mode: str,
    ) -> torch.Tensor:
        if center:
            padding = (frame_length // 2, (frame_length - 1) // 2)
        else:
            padding = (0, frame_length - 1)
        if mode != "constant" and x.dim() == 1:
            x = F.pad(x.unsqueeze(0), padding, mode=mode).squeeze(0)
        else:
            x = F.pad(x, padding, mode=mode)
        y = x.unfold(-1, frame_length, frame_period)
        if zmean:
            y = y - y.mean(-1, keepdim=True)
        return y
