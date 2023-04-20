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
import torch.nn as nn

from ..misc.utils import Lambda
from .unframe import Unframe


class InverseShortTermFourierTransform(nn.Module):
    """This is the opposite module to :func:`~diffsptk.ShortTermFourierTransform`

    Parameters
    ----------
    frame_length : int >= 1 [scalar]
        Frame length, :math:`L`.

    frame_peirod : int >= 1 [scalar]
        Frame period, :math:`P`.

    fft_length : int >= L [scalar]
        Number of FFT bins, :math:`N`.

    norm : ['none', 'power', 'magnitude']
        Normalization type of window.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular']
        Window type.

    """

    def __init__(
        self,
        frame_length,
        frame_period,
        fft_length,
        norm="power",
        window="blackman",
    ):
        super(InverseShortTermFourierTransform, self).__init__()

        self.ifft = Lambda(
            lambda x: torch.fft.irfft(x, n=fft_length)[..., :frame_length]
        )
        self.unframe = Unframe(frame_length, frame_period, norm=norm, window=window)

    def forward(self, y, out_length=None):
        """Compute inverse short-term Fourier transform.

        Parameters
        ----------
        y : Tensor [shape=(..., T/P, N/2+1)]
            Complex spectrum.

        Returns
        -------
        x : Tensor [shape=(..., T)]
            Waveform.

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
