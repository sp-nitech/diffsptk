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
from ..utils.private import check_size, filter_values, to
from .base import BaseFunctionalModule


class NegativeDerivativeOfPhaseSpectrumToCepstrum(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ndps2c.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

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
        fft_length: int,
        cep_order: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = fft_length // 2 + 1

        self.values, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("ramp", tensors[0])

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        """Convert NPDS to cepstrum.

        Parameters
        ----------
        n : Tensor [shape=(..., L/2+1)]
            The NDPS.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The cepstrum.

        Examples
        --------
        >>> n = diffsptk.ramp(4)
        >>> ndps2c = diffsptk.NegativeDerivativeOfPhaseSpectrumToCepstrum(8, 4)
        >>> c = ndps2c(n)
        >>> c
        tensor([ 0.0000, -1.7071,  0.0000, -0.0976,  0.0000])

        """
        check_size(n.size(-1), self.in_dim, "dimension of spectrum")
        return self._forward(n, *self.values, **self._buffers)

    @staticmethod
    def _func(n: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, tensors = NegativeDerivativeOfPhaseSpectrumToCepstrum._precompute(
            2 * n.size(-1) - 2, *args, **kwargs, device=n.device, dtype=n.dtype
        )
        return NegativeDerivativeOfPhaseSpectrumToCepstrum._forward(
            n, *values, *tensors
        )

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(fft_length: int, cep_order: int) -> None:
        if fft_length // 2 < max(1, cep_order):
            raise ValueError(
                "half of fft_length must be greater than or equal to cep_order."
            )
        if cep_order < 0:
            raise ValueError("cep_order must be non-negative.")

    @staticmethod
    def _precompute(
        fft_length: int,
        cep_order: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        NegativeDerivativeOfPhaseSpectrumToCepstrum._check(fft_length, cep_order)
        half_fft_length = fft_length // 2
        ramp = torch.arange(cep_order + 1, device=device, dtype=torch.double)
        ramp *= half_fft_length
        if cep_order == half_fft_length:
            ramp[-1] *= 2
        ramp[1:] = 1 / ramp[1:]
        return (cep_order,), None, (to(ramp, dtype=dtype),)

    @staticmethod
    def _forward(n: torch.Tensor, cep_order: int, ramp: torch.Tensor) -> torch.Tensor:
        c = torch.fft.hfft(n)[..., : cep_order + 1]
        c *= ramp
        return c
