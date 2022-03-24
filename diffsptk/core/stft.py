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

import torch.nn as nn

from .frame import Frame
from .spec import Spectrum
from .window import Window


class ShortTermFourierTransform(nn.Module):
    """This module is a simple cascade of framing, windowing, and spectrum calculation.

    Parameters
    ----------
    frame_length : int >= 1 [scalar]
        Frame length, :math:`L_1`.

    frame_peirod : int >= 1 [scalar]
        Frame period, :math:`P`.

    fft_length : int >= L1 [scalar]
        Number of FFT bins, :math:`L_2`.

    norm : ['none', 'power', 'magnitude']
        Normalization type of window.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
        'rectangular']
        Window type.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        Output format.

    eps : float >= 0 [scalar]
        A small value added to power spectrum.

    """

    def __init__(
        self,
        frame_length,
        frame_period,
        fft_length,
        norm="power",
        window="blackman",
        out_format="power",
        eps=1e-8,
    ):
        super(ShortTermFourierTransform, self).__init__()

        self.stft = nn.Sequential(
            Frame(frame_length, frame_period),
            Window(frame_length, fft_length, norm=norm, window=window),
            Spectrum(fft_length, out_format=out_format, eps=eps),
        )

    def forward(self, x):
        """Compute short-term Fourier transform.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Waveform.

        Returns
        -------
        y : Tensor [shape=(..., N, L/2+1)]
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
        y = self.stft(x)
        return y
