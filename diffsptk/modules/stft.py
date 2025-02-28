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

import inspect

import torch
from torch import nn

from ..misc.utils import Lambda
from ..misc.utils import get_layer
from ..misc.utils import get_values
from .base import BaseFunctionalModule
from .frame import Frame
from .spec import Spectrum
from .window import Window


class ShortTimeFourierTransform(BaseFunctionalModule):
    """This module is a simple cascade of framing, windowing, and spectrum calculation.

    Parameters
    ----------
    frame_length : int >= 1
        The frame length in samples, :math:`L`.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    fft_length : int >= L
        The number of FFT bins, :math:`N`.

    center : bool
        If True, pad the input on both sides so that the frame is centered.

    zmean : bool
        If True, perform mean subtraction on each frame.

    mode : ['constant', 'reflect', 'replicate', 'circular']
        The padding method.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular', 'nuttall']
        The window type.

    norm : ['none', 'power', 'magnitude']
        The normalization type of the window.

    eps : float >= 0
        A small value added to the power spectrum.

    relative_floor : float < 0 or None
        The relative floor of the power spectrum in dB.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power', 'complex']
        The output format.

    """

    def __init__(
        self,
        frame_length,
        frame_period,
        fft_length,
        *,
        center=True,
        zmean=False,
        mode="constant",
        window="blackman",
        norm="power",
        eps=1e-9,
        relative_floor=None,
        out_format="power",
    ):
        super().__init__()

        _, layers, _ = self._precompute(*get_values(locals()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """Compute short-time Fourier transform.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(..., T/P, N/2+1)]
            The output spectrogram.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 3)
        >>> x
        tensor([1., 2., 3.])
        >>> stft = diffsptk.STFT(frame_length=3, frame_period=1, fft_length=8)
        >>> y = stft(x)
        >>> y
        tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                [4.0000, 4.0000, 4.0000, 4.0000, 4.0000],
                [9.0000, 9.0000, 9.0000, 9.0000, 9.0000]])

        """
        return self._forward(x, *self.layers)

    @staticmethod
    def _func(x, *args, **kwargs):
        _, layers, _ = ShortTimeFourierTransform._precompute(
            *args,
            **kwargs,
        )
        return ShortTimeFourierTransform._forward(x, *layers)

    @staticmethod
    def _takes_input_size():
        return False

    @staticmethod
    def _check():
        pass

    @staticmethod
    def _precompute(
        frame_length,
        frame_period,
        fft_length,
        center,
        zmean,
        mode,
        window,
        norm,
        eps,
        relative_floor,
        out_format,
    ):
        ShortTimeFourierTransform._check()
        module = inspect.stack()[1].function == "__init__"

        frame = get_layer(
            module,
            Frame,
            dict(
                frame_length=frame_length,
                frame_period=frame_period,
                center=center,
                zmean=zmean,
                mode=mode,
            ),
        )
        window_ = get_layer(
            module,
            Window,
            dict(
                in_length=frame_length,
                out_length=fft_length,
                window=window,
                norm=norm,
            ),
        )
        if out_format == "complex":
            spec = Lambda(torch.fft.rfft)
        else:
            spec = get_layer(
                module,
                Spectrum,
                dict(
                    fft_length=fft_length,
                    eps=eps,
                    relative_floor=relative_floor,
                    out_format=out_format,
                ),
            )
        return None, (frame, window_, spec), None

    @staticmethod
    def _forward(x, frame, window, spec):
        return spec(window(frame(x)))
