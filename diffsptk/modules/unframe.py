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
from torch import nn

from ..typing import Precomputed
from ..utils.private import check_size, get_values
from .base import BaseFunctionalModule
from .window import Window


class Unframe(BaseFunctionalModule):
    """This is the opposite module to :func:`~diffsptk.Frame`.

    Parameters
    ----------
    frame_length : int >= 1
        The frame length in samples, :math:`L`.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    center : bool
        If True, pad the input on both sides so that the frame is centered.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular', 'nuttall']
        The window type.

    norm : ['none', 'power', 'magnitude']
        The normalization type of the window.

    symmetric : bool
        If True, the window is symmetric, otherwise periodic.

    learnable : bool
        Whether to make the window learnable.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        frame_length: int,
        frame_period: int,
        *,
        center: bool = True,
        window: str = "rectangular",
        norm: str = "none",
        symmetric: bool = True,
        learnable: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = frame_length

        self.values, _, tensors = self._precompute(
            *get_values(locals(), drop_keys=["learnable"])
        )
        if learnable:
            self.window = nn.Parameter(tensors[0])
        else:
            self.register_buffer("window", tensors[0])

    def forward(self, y: torch.Tensor, out_length: int | None = None) -> torch.Tensor:
        """Revert the framed waveform to the unframed waveform.

        Parameters
        ----------
        y : Tensor [shape=(..., T/P, L)]
            The framed waveform.

        out_length : int or None
            The length of the original waveform, :math:`T`.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The unframed waveform.

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
        >>> unframe = diffsptk.Unframe(5, 2)
        >>> z = unframe(y, out_length=x.size(0))
        >>> z
        tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.])

        """
        check_size(y.size(-1), self.in_dim, "length of waveform")
        return self._forward(
            y, out_length, *self.values, **self._buffers, **self._parameters
        )

    @staticmethod
    def _func(y: torch.Tensor, out_length: int | None, *args, **kwargs) -> torch.Tensor:
        values, _, tensors = Unframe._precompute(
            y.size(-1), *args, **kwargs, device=y.device, dtype=y.dtype
        )
        return Unframe._forward(y, out_length, *values, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(frame_length: int, frame_period: int) -> None:
        if frame_length <= 0:
            raise ValueError("frame_length must be positive.")
        if frame_length < frame_period:
            raise ValueError("frame_period must be less than or equal to frame_length.")

    @staticmethod
    def _precompute(
        frame_length: int,
        frame_period: int,
        center: bool = True,
        window: str = "rectangular",
        norm: str = "none",
        symmetric: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        Unframe._check(frame_length, frame_period)
        window_ = Window._precompute(
            frame_length, None, window, norm, symmetric, device=device, dtype=dtype
        )[-1][0].view(1, -1, 1)
        return (frame_length, frame_period, center), None, (window_,)

    @staticmethod
    def _forward(
        y: torch.Tensor,
        out_length: int | None,
        frame_length: int,
        frame_period: int,
        center: bool,
        window: torch.Tensor,
    ) -> torch.Tensor:
        d = y.dim()
        if d <= 1:
            raise ValueError("Input must be at least 2D tensor.")

        N = y.size(-2)

        if 4 <= d:
            batch_dims = y.size()[:-2]
            space_dims = y.size()[-2:]
            y = y.view(-1, *space_dims)

        def fold(x):
            x = F.fold(
                x,
                (1, (N - 1) * frame_period + frame_length),
                (1, frame_length),
                stride=(1, frame_period),
            )
            s = frame_length // 2 if center else 0
            e = None if out_length is None else s + out_length
            x = x[..., 0, 0, s:e]
            return x

        x = y.transpose(-2, -1)
        if d == 2:
            x = x.unsqueeze(0)
        w = window.repeat(1, 1, N)

        x = fold(x * w)
        w = fold(w * w)
        x = x / (w + 1e-16)

        if d == 2:
            x = x.squeeze(0)
        elif 4 <= d:
            x = x.view(*batch_dims, -1)
        return x
