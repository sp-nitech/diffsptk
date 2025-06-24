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

from ..utils.private import TAU, to
from .base import BaseNonFunctionalModule
from .poledf import AllPoleDigitalFilter
from .zerodf import AllZeroDigitalFilter


class GammatoneFilterBankAnalysis(BaseNonFunctionalModule):
    """Gammatone filter bank analysis module.

    Parameters
    ----------
    sample_rate : int >= 1
        The sample rate in Hz.

    f_min : float >= 0
        The minimum frequency in Hz.

    f_ref : float >= 0
        The base frequency in Hz.

    f_max : float <= sample_rate // 2
        The maximum frequency in Hz.

    filter_order : int >= 1
        The order of the Gammatone filter.

    bandwidth_factor : float > 0
        The bandwidth of the Gammatone filter.

    density : float > 0
        The density of frequencies on the ERB scale.

    exact : bool
        If False, use all-pole approximation.

    device : torch.device or None
        The device of this module.

    References
    ----------
    .. [1] V. Hohmann, "Frequency analysis and synthesis using a Gammatone filterbank,"
           *Acta Acustica united with Acustica*, vol. 88, no. 3, pp. 433-442, 2002.

    """

    def __init__(
        self,
        sample_rate: int,
        *,
        f_min: float = 70,
        f_ref: float = 1000,
        f_max: float = 6700,
        filter_order: int = 4,
        bandwidth_factor: float = 1,
        density: float = 1,
        exact: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if not (0 <= f_min <= f_ref <= f_max <= sample_rate / 2):
            raise ValueError("Invalid frequency range.")
        if filter_order <= 0:
            raise ValueError("filter_order must be positive.")
        if bandwidth_factor <= 0:
            raise ValueError("bandwidth_factor must be positive.")
        if density <= 0:
            raise ValueError("density must be positive.")

        self.exact = exact

        erb_l = 24.7
        erb_q = 9.265  # 1000 / (24.7 * 4.37)

        def hz_to_erb(hz):
            return erb_q * np.log1p(hz / (erb_l * erb_q))

        def erb_to_hz(erb):
            return (erb_l * erb_q) * np.expm1(erb / erb_q)

        # Compute center frequencies.
        erb_min = hz_to_erb(f_min)
        erb_ref = hz_to_erb(f_ref)
        erb_max = hz_to_erb(f_max)
        erb_begin = erb_ref - np.floor((erb_ref - erb_min) * density) / density
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
        lambda_ = np.exp(-TAU * b / sample_rate)
        beta = TAU * center_frequencies_in_hz / sample_rate
        z = np.exp(1j * beta)
        a_tilde = lambda_ * z

        # Compute denominator filter coefficients of Gammatone filters.
        a = np.zeros((len(a_tilde), filter_order + 1), dtype=np.complex128)
        for i in range(1, filter_order + 1):
            a[:, i] = math.comb(gamma, i) * (-a_tilde) ** i

        # Compute numerator filter coefficients of Gammatone filters.
        b = np.zeros((len(a_tilde), filter_order), dtype=np.complex128)
        if exact and 2 <= filter_order:
            ramp = np.arange(1, filter_order + 1)
            c = np.zeros(filter_order)
            c[0] = 1
            for i in range(2, filter_order):
                term1 = c * ramp
                term2 = -np.roll(term1, 1)
                term3 = i * np.roll(c, 1)
                c = term1 + term2 + term3
            b[:, 1:] = c[:-1] * a_tilde.reshape(-1, 1) ** ramp[:-1]
        else:
            b[:, 0] = 1

        # These coefficients should not be complex64 due to numerical errors.
        self.register_buffer("a", to(a, device=device, dtype=torch.complex128))
        self.register_buffer("b", to(b, device=device, dtype=torch.complex128))

        # Compute normalization factors to have 0 dB at center frequencies.
        if self.exact:
            z = to(z, device=device, dtype=torch.double)
            K = 2 / self._H(z, ignore_gain=True).diag().abs()
        else:
            a = to(a_tilde, device=device, dtype=torch.double)
            K = 2 * (1 - a.abs()) ** gamma
        K[(beta == 0) | (beta == np.pi)] *= 0.5
        self.a[:, 0] = K

        self.center_frequencies = center_frequencies_in_hz  # For synthesis.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gammatone filter banks to the input signal.

        Parameters
        ----------
        x : Tensor [shape=(B, 1, T) or (B, T) or (T,)]
            The input signal.

        Returns
        -------
        out : Tensor [shape=(B, K, T)]
            The analyzed signal.

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
        if x.dim() != 2:
            raise ValueError("Input must be 1D, 2D, or 3D tensor.")

        B, T = x.shape
        K, _ = self.a.shape

        expanded_x = x.repeat(K, 1)
        if True:
            expanded_a = self.a.repeat(B, 1).unsqueeze(1).expand(-1, T, -1)
            y = AllPoleDigitalFilter._func(expanded_x, expanded_a, frame_period=1)
        if self.exact:
            expanded_b = self.b.repeat(B, 1).unsqueeze(1).expand(-1, T, -1)
            y = AllZeroDigitalFilter._func(y, expanded_b, frame_period=1)
        y = y.reshape(K, B, T).transpose(0, 1)

        if x.dtype == torch.float:
            y = y.to(torch.complex64)
        return y

    def _H(self, z: torch.Tensor, ignore_gain: bool = False) -> torch.Tensor:
        """Return the frequency response of the filter.

        Parameters
        ----------
        z : Tensor [shape=(C,)]
            The complex frequency.

        ignore_gain : bool
            If True, the gain is ignored.

        Returns
        -------
        out : Tensor [shape=(C, K)]
            The frequency response at z for each filter.

        """
        gamma = self.a.size(-1) - 1
        K, a = torch.split(self.a, [1, gamma], dim=-1)
        if self.exact:
            ramp = torch.arange(gamma + 1, device=z.device)
            zs = z.unsqueeze(1) ** -ramp  # (C, M+1)
            numer = torch.matmul(zs[..., :-1], self.b.T)
            denom = 1 + torch.matmul(zs[..., 1:], a.T)
            F = numer / denom
        else:
            a = a[..., 0] / math.comb(gamma, 1)
            F = (1 + a.unsqueeze(0) / z.unsqueeze(1)) ** -gamma
        if not ignore_gain:
            F = K.T * F
        return F
