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

import torch.nn.functional as F

from ..utils.private import get_values
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

    """

    def __init__(
        self,
        frame_length,
        frame_period,
        *,
        center=True,
        window="rectangular",
        norm="none",
    ):
        super().__init__()

        self.values, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("window", tensors[0])

    def forward(self, y, out_length=None):
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
        return self._forward(y, out_length, *self.values, **self._buffers)

    @staticmethod
    def _func(y, out_length, *args, **kwargs):
        values, _, tensors = Unframe._precompute(
            y.size(-1), *args, **kwargs, device=y.device, dtype=y.dtype
        )
        return Unframe._forward(y, out_length, *values, *tensors)

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(frame_length, frame_period):
        if frame_length <= 0:
            raise ValueError("frame_length must be positive.")
        if frame_length < frame_period:
            raise ValueError("frame_period must be less than or equal to frame_length.")

    @staticmethod
    def _precompute(
        frame_length,
        frame_period,
        center=True,
        window="rectangular",
        norm="none",
        device=None,
        dtype=None,
    ):
        Unframe._check(frame_length, frame_period)
        window_ = Window._precompute(
            frame_length, None, window, norm, device=device, dtype=dtype
        )[-1][0].view(1, -1, 1)
        return (frame_length, frame_period, center), None, (window_,)

    @staticmethod
    def _forward(y, out_length, frame_length, frame_period, center, window):
        d = y.dim()
        N = y.size(-2)

        if not 2 <= d <= 4:
            raise ValueError("Input must be 2D, 3D, or 4D tensor.")

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

        w = window.repeat(1, 1, N)
        x = y.transpose(-2, -1)

        if d == 2:
            x = x.unsqueeze(0)

        w = fold(w)
        x = fold(x)
        x = x / w

        if d == 2:
            x = x.squeeze(0)
        return x
