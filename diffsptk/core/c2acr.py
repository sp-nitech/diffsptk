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


class CepstrumToAutocorrelation(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/c2acr.html>`_
    for details.

    Parameters
    ----------
    acr_order : int >= 0 [scalar]
        Order of autocorrelation, :math:`M_2`.

    fft_length : int >= 2 [scalar]
        Number of FFT bins, :math:`L`.

    """

    def __init__(self, acr_order, fft_length):
        super(CepstrumToAutocorrelation, self).__init__()

        self.acr_order = acr_order
        self.fft_length = fft_length

        assert 0 <= self.acr_order
        assert 2 <= self.fft_length
        assert self.acr_order <= self.fft_length // 2

    def forward(self, c):
        """Convert cepstrum to autocorrelation.

        Parameters
        ----------
        c : Tensor [shape=(..., M1+1)]
            Cepstrum.

        Returns
        -------
        r : Tensor [shape=(..., M2+1)]
            Autocorrelation.

        Examples
        --------
        >>> c = diffsptk.nrand(4)
        >>> c
        tensor([-0.1751,  0.1950, -0.3211,  0.3523, -0.5453])
        >>> c2acr = diffsptk.CepstrumToAutocorrelation(4, 16)
        >>> r = c2acr(c)
        >>> r
        tensor([ 1.0672, -0.0485, -0.1564,  0.2666, -0.4551])

        """
        x = torch.fft.rfft(c, n=self.fft_length).real
        x = torch.exp(2 * x)
        r = torch.fft.hfft(x)[..., : self.acr_order + 1]
        r = r / self.fft_length
        return r
