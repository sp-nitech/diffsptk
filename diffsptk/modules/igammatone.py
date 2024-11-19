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
from torch import nn
import torch.nn.functional as F

from ..misc.signals import impulse
from ..misc.utils import TWO_PI
from ..misc.utils import check_size
from ..misc.utils import to
from .gammatone import GammatoneFilterBankAnalysis


class GammatoneFilterBankSynthesis(nn.Module):
    """Gammatone filter bank analysis.

    Parameters
    ----------
    sample_rate : int >= 1
        Sample rate in Hz.

    desired_delay : float > 0
        Desired delay in milliseconds.

    f_min : float >= 0
        Minimum frequency in Hz.

    f_base : float >= 0
        Base frequency in Hz.

    f_max : float <= sample_rate // 2
        Maximum frequency in Hz.

    filter_order : int >= 1
        Order of Gammatone filter.

    bandwidth_factor : float > 0
        Bandwidth of Gammatone filter.

    density : float > 0
        Density of frequencies on the ERB scale.

    exact : bool
        If False, use all-pole approximation.

    n_iter : int >= 1
        Number of iterations for gain computation.

    eps : float >= 0
        Tolerance for gain computation.

    References
    ----------
    .. [1] V. Hohmann, "Frequency analysis and synthesis using a Gammatone filterbank,"
           *Acta Acustica united with Acustica*, vol. 88, no. 3, pp. 433-442, 2002.

    .. [2] T. Herzke, "Improved numerical methods for Gammatone filterbank analysis and
           synthesis," *Acta Acustica united with Acustica*, vol. 93, no. 3,
           pp. 498-500, 2007.

    """

    def __init__(
        self,
        sample_rate,
        *,
        desired_delay=4,
        f_min=70,
        f_base=1000,
        f_max=6700,
        filter_order=4,
        bandwidth_factor=1.0,
        density=1.0,
        exact=False,
        n_iter=100,
        eps=1e-8,
    ):
        super().__init__()

        self.delay = round(desired_delay * sample_rate / 1000)
        assert 1 <= self.delay, "Please increase the desired delay."
        assert 1 <= n_iter
        assert 0 <= eps

        # Compute delays.
        analyzer = GammatoneFilterBankAnalysis(
            sample_rate=sample_rate,
            f_min=f_min,
            f_base=f_base,
            f_max=f_max,
            filter_order=filter_order,
            bandwidth_factor=bandwidth_factor,
            density=density,
            exact=exact,
        )
        impulse_signal = impulse(self.delay + 1, dtype=torch.double)
        impulse_response = analyzer(impulse_signal).squeeze(0)
        max_indices = torch.argmax(impulse_response[..., :-1].abs(), dim=-1)
        row_indices = torch.arange(impulse_response.size(0))
        slopes = (
            impulse_response[row_indices, max_indices + 1]
            - impulse_response[row_indices, max_indices - 1]
        )
        slopes /= slopes.abs()
        phase_factors = 1j / slopes
        delay_samples = self.delay - max_indices

        self.register_buffer("phase_factors", to(phase_factors).unsqueeze(-1))  # (K, 1)
        self.register_buffer("delay_samples", delay_samples.unsqueeze(-1))  # (K, 1)

        # Compute gains.
        center_frequencies_in_hz = torch.from_numpy(analyzer.center_frequencies)
        z = torch.exp(1j * TWO_PI * center_frequencies_in_hz / sample_rate)
        positive_response = analyzer._H(z)
        negative_response = analyzer._H(z.conj())
        positive_response = (
            positive_response * phase_factors * z.unsqueeze(-1) ** -delay_samples
        )
        negative_response = (
            negative_response * phase_factors * z.conj().unsqueeze(-1) ** -delay_samples
        )
        combined_response = 0.5 * (positive_response + negative_response.conj())
        gains = torch.ones_like(combined_response[..., 0])
        for _ in range(n_iter):
            prev_gains = gains
            gains = gains / torch.matmul(combined_response, gains).abs()
            diff = (prev_gains - gains).abs().mean()
            if diff < eps:
                break

        self.register_buffer("gains", to(gains.real).unsqueeze(-1))  # (K, 1)

    def forward(self, y, keepdim=True, compensate_delay=True):
        """Reconstruct waveform from filter bank signals.

        Parameters
        ----------
        y : Tensor [shape=(B, K, T) or (K, T)]
            Filtered signals.

        keepdim : bool
            If True, the output shape is (B, 1, T) instead (B, T).

        compensate_delay : bool
            If True, compensate the delay.

        Returns
        -------
        out : Tensor [shape=(B, 1, T) or (B, T)]
            Reconstructed waveform.

        Examples
        --------
        >>> x = diffsptk.impulse(15999)
        >>> x[:5]
        tensor([1., 0., 0., 0., 0.])
        >>> f = diffsptk.GammatoneFilterBankAnalysis(16000)
        >>> g = diffsptk.GammatoneFilterBankSynthesis(16000)
        >>> y = g(f(x)).squeeze()
        >>> y[:5]
        tensor([ 0.8349,  0.0682, -0.1085,  0.0559, -0.0947])

        """
        if y.dim() == 2:
            y = y.unsqueeze(0)
        assert y.dim() == 3, "Input must be 3D tensor."

        B, K, T = y.shape
        check_size(K, len(self.phase_factors), "number of filters")

        # Multiply by phase factors.
        phi = self.phase_factors
        y_prime = y.real * phi.real - y.imag * phi.imag

        # Delay signals.
        max_delay = self.delay_samples.max()
        padded_y = F.pad(y_prime, (max_delay, 0))
        indices = torch.arange(T, device=y.device)
        indices = indices + max_delay - self.delay_samples
        indices = indices.unsqueeze(0).expand(B, -1, -1)
        delayed_y = padded_y.gather(-1, indices)

        # Mix signals.
        x = torch.sum(delayed_y * self.gains, dim=1, keepdim=keepdim)

        # Compensate delay.
        if compensate_delay:
            x = F.pad(x[..., self.delay :], (0, self.delay))
        return x
