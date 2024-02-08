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

from ..misc.utils import remove_gain


class Spectrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/spec.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2
        Number of FFT bins, :math:`L`.

    eps : float >= 0
        A small value added to power spectrum.

    relative_floor : float < 0 or None
        Relative floor in decibels.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        Output format.

    """

    def __init__(self, fft_length, eps=0, relative_floor=None, out_format="power"):
        super(Spectrum, self).__init__()

        assert 2 <= fft_length
        assert 0 <= eps
        assert relative_floor is None or relative_floor < 0

        self.fft_length = fft_length
        self.eps = eps
        self.const = self._precompute_const(relative_floor, out_format)

    def forward(self, b=None, a=None):
        """Compute spectrum.

        Parameters
        ----------
        b : Tensor [shape=(..., M+1)]
            Framed waveform or numerator coefficients.

        a : Tensor [shape=(..., N+1)]
            Denominator coefficients.

        Returns
        -------
        Tensor [shape=(..., L/2+1)]
            Spectrum.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 3)
        >>> x
        tensor([1., 2., 3.])
        >>> spec = diffsptk.Spectrum(8)
        >>> y = spec(x)
        >>> y
        tensor([36.0000, 25.3137,  8.0000,  2.6863,  4.0000])

        """
        return self._forward(b, a, self.fft_length, self.eps, *self.const)

    @staticmethod
    def _forward(b, a, fft_length, eps, relative_floor, out_format):
        if b is None and a is None:
            raise ValueError("Either b or a must be specified.")

        if b is not None:
            B = torch.fft.rfft(b, n=fft_length).abs()
        if a is not None:
            K, a = remove_gain(a, return_gain=True)
            A = torch.fft.rfft(a, n=fft_length).abs()

        if b is None:
            X = K / A
        elif a is None:
            X = B
        else:
            X = K * (B / A)

        s = torch.square(X) + eps
        if relative_floor is not None:
            m, _ = torch.max(s, dim=-1, keepdim=True)
            s = torch.maximum(s, m * relative_floor)
        s = out_format(s)
        return s

    @staticmethod
    def _func(b, a, fft_length, eps, relative_floor, out_format):
        const = Spectrum._precompute_const(relative_floor, out_format)
        return Spectrum._forward(b, a, fft_length, eps, *const)

    @staticmethod
    def _precompute_const(relative_floor, out_format):
        if relative_floor is None:
            r = relative_floor
        else:
            r = 10 ** (relative_floor / 10)

        if out_format == 0 or out_format == "db":
            return r, lambda x: 10 * torch.log10(x)
        elif out_format == 1 or out_format == "log-magnitude":
            return r, lambda x: 0.5 * torch.log(x)
        elif out_format == 2 or out_format == "magnitude":
            return r, lambda x: torch.sqrt(x)
        elif out_format == 3 or out_format == "power":
            return r, lambda x: x
        raise ValueError(f"out_format {out_format} is not supported.")
