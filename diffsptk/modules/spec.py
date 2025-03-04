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

from ..utils.private import get_values
from ..utils.private import remove_gain
from .base import BaseFunctionalModule


class Spectrum(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/spec.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    eps : float >= 0
        A small value added to the power spectrum.

    relative_floor : float < 0 or None
        The relative floor of the power spectrum in dB.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        The output format.

    """

    def __init__(self, fft_length, *, eps=0, relative_floor=None, out_format="power"):
        super().__init__()

        self.values = self._precompute(*get_values(locals()))

    def forward(self, b=None, a=None):
        """Compute spectrum.

        Parameters
        ----------
        b : Tensor [shape=(..., M+1)] or None
            The numerator coefficients.

        a : Tensor [shape=(..., N+1)] or None
            The denominator coefficients.

        Returns
        -------
        out : Tensor [shape=(..., L/2+1)]
            The spectrum.

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
        return self._forward(b, a, *self.values)

    @staticmethod
    def _func(b=None, a=None, *args, **kwargs):
        values = Spectrum._precompute(*args, **kwargs)
        return Spectrum._forward(b, a, *values)

    @staticmethod
    def _takes_input_size():
        return False

    @staticmethod
    def _check(fft_length, eps, relative_floor):
        if fft_length <= 1:
            raise ValueError("fft_length must be greater than 1.")
        if eps < 0:
            raise ValueError("eps must be non-negative.")
        if relative_floor is not None and 0 <= relative_floor:
            raise ValueError("relative_floor must be negative.")

    @staticmethod
    def _precompute(fft_length, eps, relative_floor, out_format):
        Spectrum._check(fft_length, eps, relative_floor)
        if relative_floor is not None:
            relative_floor = 10 ** (relative_floor / 10)
        if out_format in (0, "db"):
            formatter = lambda x: 10 * torch.log10(x)
        elif out_format in (1, "log-magnitude"):
            formatter = lambda x: 0.5 * torch.log(x)
        elif out_format in (2, "magnitude"):
            formatter = lambda x: torch.sqrt(x)
        elif out_format in (3, "power"):
            formatter = lambda x: x
        else:
            raise ValueError(f"out_format {out_format} is not supported.")
        return (fft_length, eps, relative_floor, formatter)

    @staticmethod
    def _forward(b, a, fft_length, eps, relative_floor, formatter):
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
            m = torch.amax(s, dim=-1, keepdim=True)
            s = torch.maximum(s, m * relative_floor)
        s = formatter(s)
        return s
