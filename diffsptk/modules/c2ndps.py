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
from ..utils.private import check_size, get_values, to
from .base import BaseFunctionalModule


class CepstrumToNegativeDerivativeOfPhaseSpectrum(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/c2ndps.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] B. Yegnanarayana, "Pole-zero decomposition of speech spectra," *Signal
           Processing*, vol. 3, no. 1, pp. 5-17, 1981.

    """

    def __init__(
        self,
        cep_order: int,
        fft_length: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = cep_order + 1

        self.values, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("ramp", tensors[0])

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Convert cepstrum to NDPS.

        Parameters
        ----------
        c : Tensor [shape=(..., M+1)]
            The cepstrum.

        Returns
        -------
        out : Tensor [shape=(..., L/2+1)]
            The NDPS.

        Examples
        --------
        >>> c = diffsptk.ramp(4)
        >>> c2ndps = diffsptk.CepstrumToNegativeDerivativeOfPhaseSpectrum(4, 8)
        >>> n = c2ndps(c)
        >>> n
        tensor([ 30.0000, -21.6569,  12.0000, -10.3431,  10.0000])

        """
        check_size(c.size(-1), self.in_dim, "dimension of cepstrum")
        return self._forward(c, *self.values, **self._buffers)

    @staticmethod
    def _func(c: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, tensors = CepstrumToNegativeDerivativeOfPhaseSpectrum._precompute(
            c.size(-1) - 1, *args, **kwargs, device=c.device, dtype=c.dtype
        )
        return CepstrumToNegativeDerivativeOfPhaseSpectrum._forward(
            c, *values, *tensors
        )

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(cep_order: int, fft_length: int) -> None:
        if cep_order < 0:
            raise ValueError("cep_order must be non-negative.")
        if fft_length // 2 < max(1, cep_order):
            raise ValueError(
                "half of fft_length must be greater than or equal to cep_order."
            )

    @staticmethod
    def _precompute(
        cep_order: int,
        fft_length: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        CepstrumToNegativeDerivativeOfPhaseSpectrum._check(cep_order, fft_length)
        half_fft_length = fft_length // 2
        ramp = torch.arange(cep_order + 1, device=device, dtype=torch.double) * 0.5
        if cep_order == half_fft_length:
            ramp[-1] *= 2
        return (fft_length,), None, (to(ramp, dtype=dtype),)

    @staticmethod
    def _forward(c: torch.Tensor, fft_length: int, ramp: torch.Tensor) -> torch.Tensor:
        v = c * ramp
        n = torch.fft.hfft(v, n=fft_length)[..., : fft_length // 2 + 1]
        return n
