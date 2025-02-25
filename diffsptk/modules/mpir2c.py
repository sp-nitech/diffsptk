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

from ..misc.utils import check_size
from ..misc.utils import clog
from ..misc.utils import get_values
from .c2mpir import CepstrumToMinimumPhaseImpulseResponse


class MinimumPhaseImpulseResponseToCepstrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mpir2c.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    ir_length : int >= 1
        The length of the impulse response, :math:`N`.

    n_fft : int >> N
        The number of FFT bins used for conversion. The accurate conversion requires the
        large vlaue.

    """

    def __init__(self, cep_order, ir_length, n_fft=512):
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
    def _func(h, cep_order, *args, **kwargs):
        values = MinimumPhaseImpulseResponseToCepstrum._precompute(
            cep_order, h.size(-1), *args, **kwargs
        )
        return MinimumPhaseImpulseResponseToCepstrum._forward(h, *values)

    @staticmethod
    def _check(*args, **kwargs):
        CepstrumToMinimumPhaseImpulseResponse._check(*args, **kwargs)

    @staticmethod
    def _precompute(cep_order, ir_length, n_fft):
        MinimumPhaseImpulseResponseToCepstrum._check(cep_order, ir_length, n_fft)
        return (cep_order, n_fft)

    @staticmethod
    def _forward(h, cep_order, n_fft):
        H = torch.fft.fft(h, n=n_fft)
        c = torch.fft.ifft(clog(H)).real[..., : cep_order + 1]
        c[..., 1:] *= 2
        return c
