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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Phase(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/phase.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2 [scalar]
        Number of FFT bins, :math:`L`.

    unwrap : bool [scalar]
        If True, perform phase unwrapping.

    """

    def __init__(self, fft_length, unwrap=False):
        super(Phase, self).__init__()

        self.fft_length = fft_length
        self.unwrap = unwrap

        assert 2 <= self.fft_length

    def forward(self, b, a=None):
        """Compute phase spectrum.

        Parameters
        ----------
        b : Tensor [shape=(..., M+1)]
            Numerator coefficients.

        a : Tensor [shape=(..., N+1)]
            Denominator coefficients.

        Returns
        -------
        p : Tensor [shape=(..., L/2+1)]
            Phase spectrum [:math:`\\pi` rad].

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> phase = diffsptk.Phase(8)
        >>> p = phase(x)
        >>> p
        tensor([ 0.0000, -0.5907,  0.7500, -0.1687,  1.0000])

        """
        B = torch.fft.rfft(b, n=self.fft_length)

        if a is None:
            p = torch.atan2(B.imag, B.real)
        else:
            K, a = torch.split(a, [1, a.size(-1) - 1], dim=-1)
            a = F.pad(a, (1, 0), value=1)
            A = torch.fft.rfft(a, n=self.fft_length)
            p = torch.atan2(
                B.imag * A.real - B.real * A.imag, B.real * A.real + B.imag * A.imag
            )

        # Convert to cycle [-1, 1].
        p /= math.pi

        if self.unwrap:
            diff = p[..., 1:] - p[..., :-1]
            bias = (-2 * (1 < diff)) + (2 * (diff < -1))
            s = torch.cumsum(bias, dim=-1)
            p[..., 1:] = p[..., 1:] + s

        return p
