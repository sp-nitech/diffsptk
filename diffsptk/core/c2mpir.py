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
import torch.nn as nn

from ..misc.utils import cexp
from ..misc.utils import check_size


class CepstrumToMinimumPhaseImpulseResponse(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/c2mpir.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0 [scalar]
        Order of cepstrum, :math:`M`.

    impulse_response_length : int >= 1 [scalar]
        Length of impulse response, :math:`N`.

    fft_length : int >> :math:`M` [scalar]
        Number of FFT bins, :math:`L`.

    """

    def __init__(self, cep_order, impulse_response_length, fft_length=512):
        super(CepstrumToMinimumPhaseImpulseResponse, self).__init__()

        self.cep_order = cep_order
        self.impulse_response_length = impulse_response_length
        self.fft_length = fft_length

        assert 0 <= self.cep_order
        assert 1 <= self.impulse_response_length
        assert self.impulse_response_length <= self.fft_length // 2

    def forward(self, c):
        """Convert cepstrum to minimum phase impulse response.

        Parameters
        ----------
        c : Tensor [shape=(..., M+1)]
            Cepstral coefficients.

        Returns
        -------
        h : Tensor [shape=(..., N)]
            Truncated minimum phase impulse response.

        Examples
        --------
        >>> c = diffsptk.ramp(3)
        >>> c2mpir = diffsptk.CepstrumToMinimumPhaseImpulseResponse(3, 5)
        >>> h = c2mpir(c)
        >>> h
        tensor([1.0000, 1.0000, 2.5000, 5.1667, 6.0417])

        """
        check_size(c.size(-1), self.cep_order + 1, "dimension of cepstrum")

        C = torch.fft.fft(c, n=self.fft_length)
        h = torch.fft.ifft(cexp(C))[..., : self.impulse_response_length].real
        return h
