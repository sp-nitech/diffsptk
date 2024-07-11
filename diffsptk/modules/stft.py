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

from ..misc.utils import Lambda
from .frame import Frame
from .spec import Spectrum
from .window import Window


class ShortTimeFourierTransform(nn.Module):
    """This module is a simple cascade of framing, windowing, and spectrum calculation.

    Parameters
    ----------
    frame_length : int >= 1
        Frame length, :math:`L`.

    frame_peirod : int >= 1
        Frame period, :math:`P`.

    fft_length : int >= L
        Number of FFT bins, :math:`N`.

    center : bool
        If True, assume that the center of data is the center of frame, otherwise
        assume that the center of data is the left edge of frame.

    zmean : bool
        If True, perform mean subtraction on each frame.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
        'rectangular']
        Window type.

    norm : ['none', 'power', 'magnitude']
        Normalization type of window.

    eps : float >= 0
        A small value added to power spectrum.

    relative_floor : float < 0 or None
        Relative floor in decibels.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power', 'complex']
        Output format.

    """

    def __init__(
        self,
        frame_length,
        frame_period,
        fft_length,
        *,
        center=True,
        zmean=False,
        window="blackman",
        norm="power",
        eps=1e-9,
        relative_floor=None,
        out_format="power",
    ):
        super().__init__()

        self.stft = nn.Sequential(
            Frame(frame_length, frame_period, center=center, zmean=zmean),
            Window(frame_length, fft_length, window=window, norm=norm),
            (
                Lambda(torch.fft.rfft)
                if out_format == "complex"
                else Spectrum(
                    fft_length,
                    eps=eps,
                    relative_floor=relative_floor,
                    out_format=out_format,
                )
            ),
        )

    def forward(self, x):
        """Compute short-time Fourier transform.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Waveform.

        Returns
        -------
        out : Tensor [shape=(..., T/P, N/2+1)]
            Spectrum.

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
        return self.stft(x)

    @staticmethod
    def _func(
        x,
        frame_length,
        frame_period,
        fft_length,
        center,
        zmean,
        window,
        norm,
        eps,
        relative_floor,
        out_format,
    ):
        y = Frame._func(x, frame_length, frame_period, center, zmean)
        y = Window._func(y, fft_length, window, norm)
        if out_format == "complex":
            y = torch.fft.rfft(y)
        else:
            y = Spectrum._func(y, None, fft_length, eps, relative_floor, out_format)
        return y
