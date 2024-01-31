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

from .spec import Spectrum


class AutocorrelationAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/acorr.html>`_
    for details. Currently, spectrum input is not supported.

    Parameters
    ----------
    acr_order : int >= 0 [scalar]
        Order of autocorrelation, :math:`M`.

    frame_length : int > M [scalar]
        Frame length, :math:`L`.

    norm : bool [scalar]
        If True, normalize autocorrelation.

    acf : ['none', 'biased', 'unbiased']
        Type of autocorrelation function.

    """

    def __init__(self, acr_order, frame_length, norm=False, acf="none"):
        super(AutocorrelationAnalysis, self).__init__()

        self.acr_order = acr_order
        self.norm = norm

        assert 0 <= self.acr_order
        assert self.acr_order < frame_length

        # Make spectrum module.
        fft_length = frame_length + self.acr_order
        if fft_length % 2 == 1:
            fft_length += 1
        self.spec = Spectrum(fft_length)

        # Prepare constants.
        if acf == "none":
            const = torch.tensor(1)
        elif acf == "biased":
            const = torch.tensor(frame_length)
        elif acf == "unbiased":
            const = torch.arange(frame_length, frame_length - self.acr_order - 1, -1)
        else:
            raise ValueError("acf {acf} is not supported")

        self.register_buffer("const", torch.reciprocal(const))

    def forward(self, x):
        """Estimate autocorrelation of input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Framed waveform.

        Returns
        -------
        r : Tensor [shape=(..., M+1)]
            Autocorrelation.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> acorr = diffsptk.AutocorrelationAnalysis(3, 5)
        >>> r = acorr(x)
        >>> r
        tensor([30.0000, 20.0000, 11.0000,  4.0000])

        """
        X = self.spec(x)
        r = torch.fft.irfft(X)[..., : self.acr_order + 1]
        r = r * self.const
        if self.norm:
            r = r / r[..., :1]
        return r
