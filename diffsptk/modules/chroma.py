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

from ..third_party.librosa import chroma
from ..typing import Precomputed
from ..utils.private import check_size, filter_values, to
from .base import BaseFunctionalModule


class ChromaFilterBankAnalysis(BaseFunctionalModule):
    """Chroma filter bank analysis module.

    Parameters
    ----------
    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    n_channel : int >= 1
        The number of chroma filter banks, :math:`C`.

    sample_rate : int >= 1
        The sample rate in Hz.

    norm : float
        The normalization factor.

    use_power : bool
        If True, use the power spectrum instead of the amplitude spectrum.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        *,
        fft_length: int,
        n_channel: int,
        sample_rate: int,
        norm: float = float("inf"),
        use_power: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = fft_length // 2 + 1

        self.values, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("H", tensors[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply chroma filter banks to the STFT.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            The power spectrum.

        Returns
        -------
        out : Tensor [shape=(..., C)]
            The chroma filter bank output.

        Examples
        --------
        >>> sr = 16000
        >>> x = diffsptk.sin(500, period=sr/440)
        >>> stft = diffsptk.STFT(frame_length=512, frame_period=512, fft_length=512)
        >>> chroma = diffsptk.ChromaFilterBankAnalysis(12, 512, sr)
        >>> y = chroma(stft(x))
        >>> y
        tensor([[0.1146, 0.0574, 0.0457, 0.0468, 0.0519, 0.0789, 0.1736, 0.4285, 0.7416,
                 1.0000, 0.7806, 0.3505]])

        """
        check_size(x.size(-1), self.in_dim, "dimension of spectrum")
        return self._forward(x, *self.values, **self._buffers)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, tensors = ChromaFilterBankAnalysis._precompute(
            2 * x.size(-1) - 2, *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return ChromaFilterBankAnalysis._forward(x, *values, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(fft_length: int, n_channel: int, sample_rate: int) -> None:
        if fft_length <= 1:
            raise ValueError("fft_length must be greater than 1.")
        if n_channel <= 0:
            raise ValueError("n_channel must be positive.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

    @staticmethod
    def _precompute(
        fft_length: int,
        n_channel: int,
        sample_rate: int,
        norm: float,
        use_power: bool,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        ChromaFilterBankAnalysis._check(fft_length, n_channel, sample_rate)
        H = chroma(
            sr=sample_rate,
            n_fft=fft_length,
            n_chroma=n_channel,
            base_c=True,
            dtype=np.float64,
        ).T
        H = torch.from_numpy(H)
        return (norm, use_power), None, (to(H, device=device, dtype=dtype),)

    @staticmethod
    def _forward(
        x: torch.Tensor, norm: float, use_power: bool, H: torch.Tensor
    ) -> torch.Tensor:
        y = x if use_power else torch.sqrt(x)
        y = torch.matmul(y, H)
        y = F.normalize(y, p=norm, dim=-1)
        return y
