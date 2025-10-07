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
from ..utils.private import cexp, check_size, filter_values
from .base import BaseFunctionalModule


class CepstrumToMinimumPhaseImpulseResponse(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/c2mpir.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    ir_length : int >= 1
        The length of the impulse response, :math:`N`.

    n_fft : int >> N
        The number of FFT bins used for conversion. The accurate conversion requires the
        large value.

    """

    def __init__(self, cep_order: int, ir_length: int, n_fft: int = 512) -> None:
        super().__init__()

        self.in_dim = cep_order + 1

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Convert cepstrum to minimum-phase impulse response.

        Parameters
        ----------
        c : Tensor [shape=(..., M+1)]
            The cepstral coefficients.

        Returns
        -------
        out : Tensor [shape=(..., N)]
            The truncated minimum-phase impulse response.

        Examples
        --------
        >>> import diffsptk
        >>> c2mpir = diffsptk.CepstrumToMinimumPhaseImpulseResponse(4, 5)
        >>> c = torch.tensor([0.5, -0.3, 0.2, -0.1, 0.05])
        >>> h = c2mpir(c)
        >>> h
        tensor([ 1.6487, -0.4946,  0.4039, -0.2712,  0.1803])

        """
        check_size(c.size(-1), self.in_dim, "dimension of cepstrum")
        return self._forward(c, *self.values)

    @staticmethod
    def _func(c: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = CepstrumToMinimumPhaseImpulseResponse._precompute(
            c.size(-1) - 1, *args, **kwargs
        )
        return CepstrumToMinimumPhaseImpulseResponse._forward(c, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(cep_order: int, ir_length: int, n_fft: int) -> None:
        if cep_order < 0:
            raise ValueError("cep_order must be non-negative.")
        if ir_length <= 0:
            raise ValueError("ir_length must be positive.")
        if n_fft < max(cep_order + 1, ir_length):
            raise ValueError("n_fft must be large value.")

    @staticmethod
    def _precompute(cep_order: int, ir_length: int, n_fft: int) -> Precomputed:
        CepstrumToMinimumPhaseImpulseResponse._check(cep_order, ir_length, n_fft)
        return (ir_length, n_fft)

    @staticmethod
    def _forward(c: torch.Tensor, ir_length: int, n_fft: int) -> torch.Tensor:
        C = torch.fft.fft(c, n=n_fft)
        h = torch.fft.ifft(cexp(C)).real[..., :ir_length]
        return h
