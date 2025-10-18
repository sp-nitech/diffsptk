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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..utils.private import TAU, to
from .base import BaseNonFunctionalModule


class FractionalOctaveBandAnalysis(BaseNonFunctionalModule):
    """Fractional octave filter band analysis module.

    Parameters
    ----------
    sample_rate : int >= 1
        The sample rate in Hz.

    f_min : float >= 0
        The minimum frequency in Hz.

    f_ref : float >= 0
        The reference frequency in Hz.

    f_max : float <= sample_rate // 2
        The maximum frequency in Hz.

    filter_order : int >= 3
        The order of the filter.

    n_fract : int >= 1
        The number of fractions.

    overlap : float in [0, 1]
        The overlap factor.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] J. Antoni, "Orthogonal-like fractional-octave-band filters," *The Journal of
           the Acoustical Society of America*, vol. 127, no. 2, pp. 884-895, 2010.

    """

    def __init__(
        self,
        sample_rate: int,
        *,
        f_min: float = 40,
        f_ref: float = 1000,
        f_max: float = 8000,
        filter_order: int = 1000,
        n_fract: int = 1,
        overlap: float = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if not (0 <= f_min <= f_ref <= f_max <= sample_rate / 2):
            raise ValueError("Invalid frequency range.")
        if filter_order <= 2:
            raise ValueError("filter_order must be greater than 2.")
        if n_fract <= 0:
            raise ValueError("n_fract must be positive.")
        if not (0 <= overlap <= 1):
            raise ValueError("overlap must be in [0, 1].")

        b = n_fract
        G = 10 ** (3 / 10)  # Octave frequency ratio (3 dB).

        def index_of_frequency(f):
            if b % 2 == 0:
                x = np.round(2 * b * np.log(f / f_ref) / np.log(G) - 1) / 2
            else:
                x = np.round(b * np.log(f / f_ref) / np.log(G))
            return int(x)

        def exact_center_frequency(x):
            if b % 2 == 0:
                f_m = f_ref * G ** ((2 * x + 1) / (2 * b))
            else:
                f_m = f_ref * G ** (x / b)
            return f_m

        # Compute exact center frequencies.
        x = np.arange(index_of_frequency(f_min), index_of_frequency(f_max) + 1)
        f_m = exact_center_frequency(x)
        f_m = f_m[f_m < sample_rate / 2]
        f_l = f_m * G ** (-1 / (2 * b))
        f_u = f_m * G ** (1 / (2 * b))

        # Compute indices of the center frequencies.
        c = (filter_order + 1) / sample_rate
        k_m = np.round(c * f_m).astype(int)
        k_l = np.round(c * f_l).astype(int)
        k_u = np.round(c * f_u).astype(int)

        # Compute half of the overlapping region.
        g = np.round(overlap / 2 * (k_u - k_m)).astype(int)

        # Compute magnitude response.
        magnitude = np.ones((len(f_m), (filter_order + 1) // 2 + 1))
        for j in range(1, len(f_m)):
            i = j - 1
            overlap_region = slice(k_l[j] - g[j], k_l[j] + g[j])
            magnitude[i, overlap_region.stop :] = 0
            magnitude[j, : overlap_region.start] = 0

            if 0 < g[j]:
                phi = np.arange(2 * g[j]) / (2 * g[j])
                z = np.pi / 2 * phi
                magnitude[i, overlap_region] = np.cos(z) ** 2
                magnitude[j, overlap_region] = np.sin(z) ** 2

        # Compute filter coefficients.
        freq = np.fft.rfftfreq(filter_order + 1)
        linear_phase = np.exp(-1j * TAU * filter_order / 2 * freq)
        H = magnitude * linear_phase
        h = np.fft.irfft(H)
        h = h * np.hanning(h.shape[1])
        h = np.expand_dims(h, 1)
        self.register_buffer("filters", to(h, device=device, dtype=dtype))

        # Make padding module.
        delay_left = (filter_order + 1) // 2
        delay_right = (filter_order - 1) // 2
        self.pad = nn.Sequential(
            nn.ConstantPad1d((delay_left, 0), 0),
            nn.ReplicationPad1d((0, delay_right)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform fractional octave filter band analysis.

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
        >>> import diffsptk
        >>> oband = diffsptk.FractionalOctaveBandAnalysis(16000)
        >>> x = diffsptk.ramp(0, 1, 0.25)
        >>> x
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
        >>> y = oband(x)
        >>> y.shape
        torch.Size([1, 9, 5])
        >>> y.sum(1).squeeze()
        tensor([-0.0184,  0.0969,  0.3940,  0.6062,  0.9033])

        """
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError("Input must be 1D tensor.")

        y = F.conv1d(self.pad(x), self.filters)
        return y
