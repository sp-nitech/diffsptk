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
import torch.nn.functional as F


class Spectrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/spec.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2 [scalar]
        Number of FFT bins, :math:`L`.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        Output format.

    eps : float >= 0 [scalar]
        A small value added to power spectrum.

    relative_floor : float < 0 [scalar]
        Relative floor in decibels.

    """

    def __init__(self, fft_length, out_format="power", eps=0, relative_floor=None):
        super(Spectrum, self).__init__()

        self.fft_length = fft_length
        self.eps = eps

        assert 2 <= self.fft_length
        assert 0 <= self.eps

        if relative_floor is None:
            self.relative_floor = None
        else:
            assert relative_floor < 0
            self.relative_floor = 10 ** (relative_floor / 10)

        if out_format == 0 or out_format == "db":
            self.convert = lambda x: 10 * torch.log10(x)
        elif out_format == 1 or out_format == "log-magnitude":
            self.convert = lambda x: 0.5 * torch.log(x)
        elif out_format == 2 or out_format == "magnitude":
            self.convert = lambda x: torch.sqrt(x)
        elif out_format == 3 or out_format == "power":
            self.convert = lambda x: x
        else:
            raise ValueError(f"out_format {out_format} is not supported")

    def forward(self, b, a=None):
        """Convert waveform to spectrum.

        Parameters
        ----------
        b : Tensor [shape=(..., M+1)]
            Framed waveform or numerator coefficients.

        a : Tensor [shape=(..., N+1)]
            Denominator coefficients.

        Returns
        -------
        y : Tensor [shape=(..., L/2+1)]
            Spectrum.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 3)
        >>> x
        tensor([1., 2., 3.])
        >>> spec = diffsptk.Spectrum(fft_length=8)
        >>> y = spec(x)
        >>> y
        tensor([36.0000, 25.3137,  8.0000,  2.6863,  4.0000])

        """
        X = torch.fft.rfft(b, n=self.fft_length).abs()

        if a is not None:
            K, a1 = torch.split(a, [1, a.size(-1) - 1], dim=-1)
            a = F.pad(a1, (1, 0), value=1)
            X /= torch.fft.rfft(a, n=self.fft_length).abs()
            X *= K

        y = torch.square(X) + self.eps
        if self.relative_floor is not None:
            m, _ = torch.max(y, dim=-1, keepdim=True)
            y = torch.maximum(y, m * self.relative_floor)
        y = self.convert(y)
        return y
