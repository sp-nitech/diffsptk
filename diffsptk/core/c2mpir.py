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
    for details. The conversion uses FFT instead of recursive formula.

    Parameters
    ----------
    cep_order : int >= 0 [scalar]
        Order of cepstrum, :math:`M`.

    ir_length : int >= 1 [scalar]
        Length of impulse response, :math:`N`.

    n_fft : int >> :math:`N` [scalar]
        Number of FFT bins. Accurate conversion requires the large value.

    """

    def __init__(self, cep_order, ir_length, n_fft=512):
        super(CepstrumToMinimumPhaseImpulseResponse, self).__init__()

        self.cep_order = cep_order
        self.ir_length = ir_length
        self.n_fft = n_fft

        assert 0 <= self.cep_order
        assert 1 <= self.ir_length
        assert max(self.cep_order + 1, self.ir_length) < self.n_fft

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

        C = torch.fft.fft(c, n=self.n_fft)
        h = torch.fft.ifft(cexp(C))[..., : self.ir_length].real
        return h
