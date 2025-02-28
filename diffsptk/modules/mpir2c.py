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

from ..misc.utils import check_size
from ..misc.utils import clog
from ..misc.utils import get_values
from .base import BaseFunctionalModule


class MinimumPhaseImpulseResponseToCepstrum(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mpir2c.html>`_
    for details.

    Parameters
    ----------
    ir_length : int >= 1
        The length of the impulse response, :math:`N`.

    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    n_fft : int >> N
        The number of FFT bins used for conversion. The accurate conversion requires the
        large value.

    """

    def __init__(self, ir_length, cep_order, n_fft=512):
        super().__init__()

        self.in_dim = ir_length

        self.values = self._precompute(*get_values(locals()))

    def forward(self, h):
        """Convert minimum-phase impulse response to cepstrum.

        Parameters
        ----------
        h : Tensor [shape=(..., N)]
            The truncated minimum-phase impulse response.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The cepstral coefficients.

        Examples
        --------
        >>> h = diffsptk.ramp(4, 0, -1)
        >>> mpir2c = diffsptk.MinimumPhaseImpulseResponseToCepstrum(3, 5)
        >>> c = mpir2c(h)
        >>> c
        tensor([1.3863, 0.7500, 0.2188, 0.0156])

        """
        check_size(h.size(-1), self.in_dim, "impulse response length")
        return self._forward(h, *self.values)

    @staticmethod
    def _func(h, *args, **kwargs):
        values = MinimumPhaseImpulseResponseToCepstrum._precompute(
            h.size(-1), *args, **kwargs
        )
        return MinimumPhaseImpulseResponseToCepstrum._forward(h, *values)

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(ir_length, cep_order, n_fft):
        if ir_length <= 0:
            raise ValueError("ir_length must be positive.")
        if cep_order < 0:
            raise ValueError("cep_order must be non-negative.")
        if n_fft < max(cep_order + 1, ir_length):
            raise ValueError("n_fft must be large value.")

    @staticmethod
    def _precompute(ir_length, cep_order, n_fft):
        MinimumPhaseImpulseResponseToCepstrum._check(ir_length, cep_order, n_fft)
        return (cep_order, n_fft)

    @staticmethod
    def _forward(h, cep_order, n_fft):
        H = torch.fft.fft(h, n=n_fft)
        c = torch.fft.ifft(clog(H)).real[..., : cep_order + 1]
        c[..., 1:] *= 2
        return c
