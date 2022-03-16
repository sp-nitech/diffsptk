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


class Spectrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/spec.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2 [scalar]
        Number of FFT bins, :math:`L`.

    out_format : ['db', 'logamp', 'amp', 'pow']
        Output format.

    eps : float >= 0 [scalar]
        A small value added to power spectrum.

    """

    def __init__(self, fft_length, out_format="power", eps=1e-8):
        super(Spectrum, self).__init__()

        self.fft_length = fft_length
        self.eps = eps

        assert 2 <= self.fft_length
        assert 0 <= self.eps

        if out_format == 0 or out_format == "db":
            self.convert = lambda x: 10 * torch.log10(x)
        elif out_format == 1 or out_format == "log":
            self.convert = lambda x: 0.5 * torch.log(x)
        elif out_format == 2 or out_format == "magnitude":
            self.convert = lambda x: torch.sqrt(x)
        elif out_format == 3 or out_format == "power":
            self.convert = lambda x: x
        else:
            raise ValueError(f"out_format {out_format} is not supported")

    def forward(self, x):
        """Convert waveform to spectrum.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Framed waveform.

        Returns
        -------
        y : Tensor [shape=(..., L/2+1)]
            Spectrum.

        """
        X = torch.fft.rfft(x, n=self.fft_length)
        y = torch.square(torch.abs(X)) + self.eps
        y = self.convert(y)
        return y
