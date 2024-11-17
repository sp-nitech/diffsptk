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

import numpy as np
import torch
from torch import nn

from ..misc.utils import TWO_PI
from .poledf import AllPoleDigitalFilter


class GammatoneFilterBankAnalysis(nn.Module):
    """Gammatone filter bank analysis.

    Parameters
    ----------
    sample_rate : int >= 1
        Sample rate in Hz.

    f_min : float >= 0
        Minimum frequency in Hz.

    f_base : float > 0
        Base frequency in Hz.

    f_max : float <= sample_rate // 2
        Maximum frequency in Hz.

    filter_order : int >= 1
        Order of Gammatone filter.

    bandwidth_factor : float > 0
        Bandwidth of Gammatone filter.

    density : float > 0
        Density of frequencies on the ERB scale.

    References
    ----------
    .. [1] V. Hohmann, "Frequency analysis and synthesis using a Gammatone filterbank,"
           *Acta Acustica united with Acustica*, vol. 88, no. 3, pp. 433-442, 2002.

    """

    def __init__(
        self,
        sample_rate,
        *,
        f_min=70,
        f_base=1000,
        f_max=6700,
        filter_order=4,
        bandwidth_factor=1.0,
        density=1.0,
    ):
        super().__init__()

        assert 0 <= f_min <= f_base <= f_max < sample_rate / 2
        assert 1 <= filter_order
        assert 0 < bandwidth_factor
        assert 0 < density

        erb_l = 24.7
        erb_q = 9.265

        def hz_to_erb(hz):
            return erb_q * np.log1p(hz / (erb_l * erb_q))

        def erb_to_hz(erb):
            return (erb_l * erb_q) * np.expm1(erb / erb_q)

        # Compute center frequencies.
        erb_min = hz_to_erb(f_min)
        erb_base = hz_to_erb(f_base)
        erb_max = hz_to_erb(f_max)
        erb_begin = erb_base - np.floor((erb_base - erb_min) * density) / density
        center_frequencies = np.arange(erb_begin, erb_max + 1e-6, 1 / density)
        center_frequencies_in_hz = erb_to_hz(center_frequencies)

        # Compute filter coefficients of 1st-order all-pole filters.
        erb_audiological = (erb_l + center_frequencies_in_hz / erb_q) * bandwidth_factor
        gamma = filter_order
        a_gamma = (
            np.pi
            * math.factorial(2 * gamma - 2)
            * (2 ** -(2 * gamma - 2))
            / math.factorial(gamma - 1) ** 2
        )
        b = erb_audiological / a_gamma
        lambda_ = np.exp(-TWO_PI * b / sample_rate)
        beta = TWO_PI * center_frequencies_in_hz / sample_rate
        a_tilde = lambda_ * np.exp(1j * beta)
        K = 2 * (1 - np.abs(a_tilde)) ** gamma
        K[(beta == 0) | (beta == np.pi)] *= 0.5

        # Compute filter coefficients of Gammatone filters.
        a = np.zeros((len(a_tilde), filter_order + 1), dtype=np.complex128)
        a[:, 0] = K
        for i in range(1, filter_order + 1):
            a[:, i] = math.comb(gamma, i) * (-a_tilde) ** i

        self.register_buffer("a", torch.from_numpy(a))  # This should not be float32.
        self.center_frequencies = center_frequencies_in_hz  # For synthesis.

    def forward(self, x):
        """Apply Gammatone filter banks to signals.

        Parameters
        ----------
        x : Tensor [shape=(B, 1, T) or (B, T) or (T,)]
            Original waveform.

        Returns
        -------
        out : Tensor [shape=(B, K, T)]
            Filtered signals.

        Examples
        --------
        >>> x = diffsptk.impulse(15999)
        >>> gammatone = diffsptk.GammatoneFilterBankAnalysis(16000)
        >>> y = gammatone(x)
        >>> y.shape
        torch.Size([1, 30, 16000])

        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            x = x.squeeze(1)
        assert x.dim() == 2, "Input must be 2D tensor."

        B, T = x.shape
        K, _ = self.a.shape

        expanded_x = x.repeat(K, 1)
        expanded_a = self.a.repeat(B, 1).unsqueeze(1).expand(-1, T, -1)
        y = AllPoleDigitalFilter._forward(expanded_x, expanded_a, frame_period=1)
        y = y.reshape(K, B, T).transpose(0, 1)

        if x.dtype == torch.float:
            y = y.to(torch.complex64)
        return y

    def _H(self, z):
        gamma = self.a.size(-1) - 1
        K = self.a[..., 0]
        a = self.a[..., 1] / math.comb(gamma, 1)
        return K * (1 + a.unsqueeze(0) / z.unsqueeze(1)) ** -gamma
