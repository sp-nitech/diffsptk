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

from ..typing import Callable, Precomputed
from ..utils.private import check_size, get_values, to
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

    use_power : bool
        If True, use the power spectrum instead of the amplitude spectrum.

    out_format : ['y', 'yE', 'y,E']
        `y` is mel filber bank output and `E` is energy. If this is `yE`, the two output
        tensors are concatenated and return the tensor instead of the tuple.

    References
    ----------
    .. [1] S. Young et al., "The HTK Book," *Cambridge University Press*, 2006.

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
        use_power: bool = False,
        out_format: str | int = "y",
    ) -> None:
        super().__init__()

        self.in_dim = fft_length // 2 + 1

        self.values, _, tensors = self._precompute(*get_values(locals()))
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
        return self._forward(x, *self.values, **self._buffers)

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

    @staticmethod
    def _precompute(
        fft_length: int,
        n_channel: int,
        sample_rate: int,
        f_min: float,
        f_max: float | None,
        floor: float,
        gamma: float,
        use_power: bool,
        out_format: str | int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        MelFilterBankAnalysis._check(
            fft_length,
            n_channel,
            sample_rate,
            f_min,
            f_max,
            floor,
            gamma,
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

        def hz_to_mel(x):
            return 1127 * np.log1p(x / 700)

        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)

        lower_bin_index = max(1, int(f_min / sample_rate * fft_length + 1.5))
        upper_bin_index = min(
            fft_length // 2, int(f_max / sample_rate * fft_length + 0.5)
        )

        seed = np.arange(1, n_channel + 2)
        center_frequencies = (mel_max - mel_min) / (n_channel + 1) * seed + mel_min

        bin_indices = np.arange(lower_bin_index, upper_bin_index)
        mel = hz_to_mel(sample_rate * bin_indices / fft_length)
        lower_channel_map = [np.argmax(0 < (m <= center_frequencies)) for m in mel]

        diff = center_frequencies - np.insert(center_frequencies[:-1], 0, mel_min)
        weights = torch.zeros(
            (fft_length // 2 + 1, n_channel), device=device, dtype=torch.double
        )
        for i, k in enumerate(bin_indices):
            m = lower_channel_map[i]
            w = (center_frequencies[max(0, m)] - mel[i]) / diff[max(0, m)]
            if 0 < m:
                weights[k, m - 1] += w
            if m < n_channel:
                weights[k, m] += 1 - w

        return (floor, gamma, use_power, formatter), None, (to(weights, dtype=dtype),)

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
        y = torch.log(y) if gamma == 0 else (torch.pow(x, gamma) - 1) / gamma
        E = (2 * x[..., 1:-1]).sum(-1) + x[..., 0] + x[..., -1]
        E = torch.log(E / (2 * (x.size(-1) - 1))).unsqueeze(-1)
        return formatter(y, E)
