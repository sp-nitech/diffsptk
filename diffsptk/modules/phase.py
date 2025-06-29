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

from ..typing import Precomputed
from ..utils.private import filter_values, remove_gain
from .base import BaseFunctionalModule


class Phase(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/phase.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    unwrap : bool
        If True, perform the phase unwrapping.

    """

    def __init__(self, fft_length: int, unwrap: bool = False) -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(
        self, b: torch.Tensor | None = None, a: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute phase spectrum.

        Parameters
        ----------
        b : Tensor [shape=(..., M+1)] or None
            The numerator coefficients.

        a : Tensor [shape=(..., N+1)] or None
            The denominator coefficients.

        Returns
        -------
        out : Tensor [shape=(..., L/2+1)]
            The phase spectrum [:math:`\\pi` rad].

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> phase = diffsptk.Phase(8)
        >>> p = phase(x)
        >>> p
        tensor([ 0.0000, -0.5907,  0.7500, -0.1687,  1.0000])

        """
        return self._forward(b, a, *self.values)

    @staticmethod
    def _func(b: torch.Tensor, a: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = Phase._precompute(*args, **kwargs)
        return Phase._forward(b, a, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(fft_length: int) -> None:
        if fft_length <= 1:
            raise ValueError("fft_length must be greater than 1.")

    @staticmethod
    def _precompute(fft_length: int, unwrap: bool) -> Precomputed:
        Phase._check(fft_length)
        return (fft_length, unwrap)

    @staticmethod
    def _forward(
        b: torch.Tensor | None,
        a: torch.Tensor | None,
        fft_length: int,
        unwrap: bool,
    ) -> torch.Tensor:
        if b is None and a is None:
            raise ValueError("Either b or a must be specified.")

        if b is not None:
            B = torch.fft.rfft(b, n=fft_length)
        if a is not None:
            A = torch.fft.rfft(remove_gain(a), n=fft_length)

        if b is None:
            numer = -A.imag
            denom = A.real
        elif a is None:
            numer = B.imag
            denom = B.real
        else:
            numer = B.imag * A.real - B.real * A.imag
            denom = B.real * A.real + B.imag * A.imag
        p = torch.atan2(numer, denom)

        # Convert to cycle [-1, 1].
        p /= torch.pi

        if unwrap:
            diff = torch.diff(p, dim=-1)
            bias = (-2 * (1 < diff)) + (2 * (diff < -1))
            s = torch.cumsum(bias, dim=-1)
            p[..., 1:] += s
        return p
