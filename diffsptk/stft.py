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

from diffsptk.frame import Frame
from diffsptk.spec import Spectrum
from diffsptk.window import Window


class STFT(nn.Module):
    """This module is a simple cascade of framing, windowing, and spectrum calculation.

    See also
    --------
    diffsptk.frame, diffsptk.window, diffsptk.spec

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
        super(STFT, self).__init__()

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

        """
        y = self.stft(x)
        return y
