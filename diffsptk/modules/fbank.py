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
from torch import nn

from ..typing import Callable, Precomputed
from ..utils.private import (
    auditory_to_hz,
    check_size,
    filter_values,
    hz_to_auditory,
    to,
)
from .base import BaseFunctionalModule


class MelFilterBankAnalysis(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/fbank.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    n_channel : int >= 1
        The number of mel filter banks, :math:`C`.

    sample_rate : int >= 1
        The sample rate in Hz.

    f_min : float >= 0
        The minimum frequency in Hz.

    f_max : float <= sample_rate // 2
        The maximum frequency in Hz.

    floor : float > 0
        The minimum mel filter bank output in linear scale.

    gamma : float in [-1, 1]
        The parameter of the generalized logarithmic function.

    scale : ['htk', 'mel', 'inverted-mel', 'bark', 'linear']
        The type of auditory scale used to construct the filter bank.

    erb_factor : float > 0 or None
        The scale factor for the ERB scale, referred to as the E-factor. If not None,
        the filter bandwidths are adjusted according to the scaled ERB scale.

    use_power : bool
        If True, use the power spectrum instead of the amplitude spectrum.

    out_format : ['y', 'yE', 'y,E']
        `y` is mel filber bank output and `E` is energy. If this is `yE`, the two output
        tensors are concatenated and return the tensor instead of the tuple.

    learnable : bool
        Whether to make the basis learnable.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] S. Young et al., "The HTK Book Version 3.4," *Cambridge University Press*,
           2006.

    .. [2] T. Ganchev et al., "Comparative evaluation of various MFCC implementations on
           the speaker verification task," *Proceedings of SPECOM*, vol. 1, pp. 191-194,
           2005.

    .. [3] M. D. Skowronski et al., "Exploiting independent filter bandwidth of human
           factor cepstral coefficients in automatic speech recognition," *The Journal
           of the Acoustical Society of America*, vol. 116, no. 3, pp. 1774-1780, 2004.

    """

    def __init__(
        self,
        *,
        fft_length: int,
        n_channel: int,
        sample_rate: int,
        f_min: float = 0,
        f_max: float | None = None,
        floor: float = 1e-5,
        gamma: float = 0,
        scale: str = "htk",
        erb_factor: float | None = None,
        use_power: bool = False,
        out_format: str | int = "y",
        learnable: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = fft_length // 2 + 1

        self.values, _, tensors = self._precompute(
            **filter_values(locals(), drop_keys=["learnable"])
        )
        if learnable:
            self.H = nn.Parameter(tensors[0])
        else:
            self.register_buffer("H", tensors[0])

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply mel filter banks to the STFT.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            The power spectrum.

        Returns
        -------
        y : Tensor [shape=(..., C)]
            The mel filter bank output.

        E : Tensor [shape=(..., 1)] (optional)
            The energy.

        Examples
        --------
        >>> x = diffsptk.ramp(19)
        >>> stft = diffsptk.STFT(frame_length=10, frame_period=10, fft_length=32)
        >>> fbank = diffsptk.MelFilterBankAnalysis(
        ...     fft_length=32, n_channel=4, sample_rate=8000
        ... )
        >>> y = fbank(stft(x))
        >>> y
        tensor([[0.1214, 0.4825, 0.6072, 0.3589],
                [3.3640, 3.4518, 2.7717, 0.5088]])

        """
        check_size(x.size(-1), self.in_dim, "dimension of spectrum")
        return self._forward(x, *self.values, **self._buffers, **self._parameters)

    @staticmethod
    def _func(
        x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        values, _, tensors = MelFilterBankAnalysis._precompute(
            2 * x.size(-1) - 2, *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return MelFilterBankAnalysis._forward(x, *values, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(
        fft_length: int,
        n_channel: int,
        sample_rate: int,
        f_min: float,
        f_max: float | None,
        floor: float,
        gamma: float,
        erb_factor: float | None,
    ) -> None:
        if fft_length <= 1:
            raise ValueError("fft_length must be greater than 1.")
        if n_channel <= 0:
            raise ValueError("n_channel must be positive.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if f_min < 0 or sample_rate / 2 <= f_min:
            raise ValueError("Invalid f_min.")
        if f_max is not None and not (f_min < f_max <= sample_rate / 2):
            raise ValueError("Invalid f_min and f_max.")
        if floor <= 0:
            raise ValueError("floor must be positive.")
        if 1 < abs(gamma):
            raise ValueError("gamma must be in [-1, 1].")
        if erb_factor is not None and erb_factor <= 0:
            raise ValueError("erb_factor must be positive.")

    @staticmethod
    def _precompute(
        fft_length: int,
        n_channel: int,
        sample_rate: int,
        f_min: float,
        f_max: float | None,
        floor: float,
        gamma: float,
        scale: str,
        erb_factor: float | None,
        use_power: bool,
        out_format: str | int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        MelFilterBankAnalysis._check(
            fft_length,
            n_channel,
            sample_rate,
            f_min,
            f_max,
            floor,
            gamma,
            erb_factor,
        )

        if out_format in (0, "y"):
            formatter = lambda y, E: y
        elif out_format in (1, "yE"):
            formatter = lambda y, E: torch.cat((y, E), dim=-1)
        elif out_format in (2, "y,E"):
            formatter = lambda y, E: (y, E)
        else:
            raise ValueError(f"out_format {out_format} is not supported.")

        if f_max is None:
            f_max = sample_rate / 2

        weights = np.zeros((fft_length // 2 + 1, n_channel))

        if erb_factor is None:
            mel_min = hz_to_auditory(f_min, scale)
            mel_max = hz_to_auditory(f_max, scale)

            lower_bin_index = max(1, int(f_min / sample_rate * fft_length + 1.5))
            upper_bin_index = min(
                fft_length // 2, int(f_max / sample_rate * fft_length + 0.5)
            )

            seed = np.arange(1, n_channel + 2)
            center_frequencies = (mel_max - mel_min) / (n_channel + 1) * seed + mel_min

            bin_indices = np.arange(lower_bin_index, upper_bin_index)
            mel = hz_to_auditory(sample_rate * bin_indices / fft_length, scale)
            lower_channel_map = [np.argmax(0 < (m <= center_frequencies)) for m in mel]
            diff = center_frequencies - np.insert(center_frequencies[:-1], 0, mel_min)
            for i, k in enumerate(bin_indices):
                m = lower_channel_map[i]
                w = (center_frequencies[max(0, m)] - mel[i]) / diff[max(0, m)]
                if 0 < m:
                    weights[k, m - 1] = w
                if m < n_channel:
                    weights[k, m] = 1 - w
        else:
            a = erb_factor * 6.23e-6
            b = erb_factor * 93.39e-3
            c = erb_factor * 28.52

            def compute_center_frequency(f, at_first):
                sign = 1 if at_first else -1
                a_hat = sign * 0.5 * (1 / (700 + f))
                b_hat = sign * 700 / (700 + f)
                c_hat = -sign * 0.5 * f * (1 + 700 / (700 + f))

                b_bar = (b - b_hat) / (a - a_hat)
                c_bar = (c - c_hat) / (a - a_hat)
                return 0.5 * (-b_bar + np.sqrt(b_bar**2 - 4 * c_bar))

            fc_1 = compute_center_frequency(f_min, True)
            fc_C = compute_center_frequency(f_max, False)
            zc_1 = hz_to_auditory(fc_1, scale)
            zc_C = hz_to_auditory(fc_C, scale)
            fc = auditory_to_hz(np.linspace(zc_1, zc_C, n_channel), scale)
            erb = a * fc**2 + b * fc + c
            # Note that the equation (C3) in the original paper is incorrect.
            fl = -(700 + erb) + np.sqrt(erb**2 + (700 + fc) ** 2)
            fh = fl + 2 * erb

            f = np.linspace(0, sample_rate / 2, fft_length // 2 + 1)
            for m, (low, center, high) in enumerate(zip(fl, fc, fh)):
                mask = (low <= f) & (f < center)
                weights[mask, m] = (f[mask] - low) / (center - low)
                mask = (center <= f) & (f <= high)
                weights[mask, m] = (high - f[mask]) / (high - center)

        weights = torch.from_numpy(weights)

        return (
            (floor, gamma, use_power, formatter),
            None,
            (to(weights, device=device, dtype=dtype),),
        )

    @staticmethod
    def _forward(
        x: torch.Tensor,
        floor: float,
        gamma: float,
        use_power: bool,
        formatter: Callable,
        H: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        y = x if use_power else torch.sqrt(x)
        y = torch.matmul(y, H)
        y = torch.clip(y, min=floor)
        y = torch.log(y) if gamma == 0 else ((torch.pow(y, gamma) - 1) / gamma)
        E = (2 * x[..., 1:-1]).sum(-1) + x[..., 0] + x[..., -1]
        E = torch.log(E / (2 * (x.size(-1) - 1))).unsqueeze(-1)
        return formatter(y, E)
