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
from ..utils.private import check_size, filter_values
from .base import BaseFunctionalModule


class CepstrumToAutocorrelation(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/c2acr.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    acr_order : int >= 0
        The order of the autocorrelation, :math:`N`.

    n_fft : int >> N
        The number of FFT bins used for conversion. The accurate conversion requires the
        large value.

    """

    def __init__(self, cep_order: int, acr_order: int, n_fft: int = 512) -> None:
        super().__init__()

        self.in_dim = cep_order + 1

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Convert cepstrum to autocorrelation.

        Parameters
        ----------
        c : Tensor [shape=(..., M+1)]
            The cepstral coefficients.

        Returns
        -------
        out : Tensor [shape=(..., N+1)]
            The autocorrelation.

        Examples
        --------
        >>> c = diffsptk.nrand(4)
        >>> c
        tensor([-0.1751,  0.1950, -0.3211,  0.3523, -0.5453])
        >>> c2acr = diffsptk.CepstrumToAutocorrelation(4, 4, 16)
        >>> r = c2acr(c)
        >>> r
        tensor([ 1.0672, -0.0485, -0.1564,  0.2666, -0.4551])

        """
        check_size(c.size(-1), self.in_dim, "dimension of cepstrum")
        return self._forward(c, *self.values)

    @staticmethod
    def _func(c: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = CepstrumToAutocorrelation._precompute(c.size(-1) - 1, *args, **kwargs)
        return CepstrumToAutocorrelation._forward(c, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(cep_order: int, acr_order: int, n_fft: int) -> None:
        if cep_order < 0:
            raise ValueError("cep_order must be non-negative.")
        if acr_order < 0:
            raise ValueError("acr_order must be non-negative.")
        if n_fft < max(cep_order + 1, acr_order + 1):
            raise ValueError("n_fft must be large value.")

    @staticmethod
    def _precompute(cep_order: int, acr_order: int, n_fft: int) -> Precomputed:
        CepstrumToAutocorrelation._check(cep_order, acr_order, n_fft)
        return (acr_order, n_fft)

    @staticmethod
    def _forward(c: torch.Tensor, acr_order: int, n_fft: int) -> torch.Tensor:
        x = torch.fft.rfft(c, n=n_fft).real
        x = torch.exp(2 * x)
        r = torch.fft.hfft(x, norm="forward")[..., : acr_order + 1]
        return r
