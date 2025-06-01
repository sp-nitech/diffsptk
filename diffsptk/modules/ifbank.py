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

from ..typing import Precomputed
from ..utils.private import check_size, get_values
from .base import BaseFunctionalModule
from .fbank import MelFilterBankAnalysis


class InverseMelFilterBankAnalysis(BaseFunctionalModule):
    """This is the opposite module to :func:~diffsptk.MelFilterBankAnalysis`.

    Parameters
    ----------
    n_channel : int >= 1
        The number of mel filter banks, :math:`C`.

    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    sample_rate : int >= 1
        The sample rate in Hz.

    f_min : float >= 0
        The minimum frequency in Hz.

    f_max : float <= sample_rate // 2
        The maximum frequency in Hz.

    gamma : float in [-1, 1]
        The parameter of the generalized logarithmic function.

    use_power : bool
        Set to True if the mel filter bank output is extracted from the power spectrum
        instead of the amplitude spectrum.

    learnable : bool
        Whether to make the basis learnable.

    """

    def __init__(
        self,
        *,
        n_channel: int,
        fft_length: int,
        sample_rate: int,
        f_min: float = 0,
        f_max: float | None = None,
        gamma: float = 0,
        use_power: bool = False,
        learnable: bool = False,
    ) -> None:
        super().__init__()

        self.in_dim = n_channel

        self.values, _, tensors = self._precompute(*get_values(locals(), drop=1))
        if learnable:
            self.H = nn.Parameter(tensors[0])
        else:
            self.register_buffer("H", tensors[0])

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Reconstruct the power spectrum from the mel filter bank output.

        Parameters
        ----------
        y : Tensor [shape=(..., C)]
            The mel filter bank output.

        Returns
        -------
        out : Tensor [shape=(..., L/2+1)]
            The power spectrum.

        Examples
        --------

        """
        check_size(y.size(-1), self.in_dim, "dimension of mel spectrogram")
        return self._forward(y, *self.values, **self._buffers, **self._parameters)

    @staticmethod
    def _func(y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, tensors = InverseMelFilterBankAnalysis._precompute(
            y.size(-1), *args, **kwargs, device=y.device, dtype=y.dtype
        )
        return InverseMelFilterBankAnalysis._forward(y, *values, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check() -> None:
        pass

    @staticmethod
    def _precompute(
        n_channel: int,
        fft_length: int,
        sample_rate: int,
        f_min: float,
        f_max: float | None,
        gamma: float,
        use_power: bool,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        InverseMelFilterBankAnalysis._check()

        _, _, H = MelFilterBankAnalysis._precompute(
            fft_length,
            n_channel,
            sample_rate,
            f_min,
            f_max,
            device=device,
            dtype=dtype,
        )
        H = H[0].pinverse()

        return (gamma, use_power), None, (H,)

    @staticmethod
    def _forward(
        y: torch.Tensor,
        gamma: float,
        use_power: bool,
        H: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.exp(y) if gamma == 0 else torch.pow(gamma * y + 1, 1 / gamma)
        x = torch.matmul(x, H)
        x = x if use_power else torch.square(x)
        return x
