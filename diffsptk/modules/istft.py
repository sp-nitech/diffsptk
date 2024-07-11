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
from .unframe import Unframe


class InverseShortTimeFourierTransform(nn.Module):
    """This is the opposite module to :func:`~diffsptk.ShortTimeFourierTransform`.

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

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular']
        Window type.

    norm : ['none', 'power', 'magnitude']
        Normalization type of window.

    """

    def __init__(
        self,
        frame_length,
        frame_period,
        fft_length,
        *,
        center=True,
        window="blackman",
        norm="power",
    ):
        super().__init__()

        self.ifft = Lambda(
            lambda x: torch.fft.irfft(x, n=fft_length)[..., :frame_length]
        )
        self.unframe = Unframe(
            frame_length, frame_period, center=center, norm=norm, window=window
        )

    def forward(self, y, out_length=None):
        """Compute inverse short-time Fourier transform.

        Parameters
        ----------
        y : Tensor [shape=(..., T/P, N/2+1)]
            Complex spectrum.

        out_length : int or None
            Length of output waveform.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            Reconstructed waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 3)
        >>> x
        tensor([1., 2., 3.])
        >>> stft_params = {"frame_length": 3, "frame_period": 1, "fft_length": 8}
        >>> stft = diffsptk.STFT(**stft_params, out_format="complex")
        >>> istft = diffsptk.ISTFT(**stft_params)
        >>> y = istft(stft(x), out_length=3)
        >>> y
        tensor([1., 2., 3.])

        """
        x = self.ifft(y)
        x = self.unframe(x, out_length=out_length)
        return x

    @staticmethod
    def _func(
        y, out_length, frame_length, frame_period, fft_length, center, window, norm
    ):
        x = torch.fft.irfft(y, n=fft_length)[..., :frame_length]
        x = Unframe._func(
            x,
            out_length,
            frame_length,
            frame_period,
            center=center,
            window=window,
            norm=norm,
        )
        return x
