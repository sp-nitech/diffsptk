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


class Autocorrelation(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/acorr.html>`_
    for details.

    Parameters
    ----------
    frame_length : int > M
        Frame length, :math:`L`.

    acr_order : int >= 0
        Order of autocorrelation, :math:`M`.

    norm : bool
        If True, normalize the autocorrelation.

    estimator : ['none', 'biased', 'unbiased']
        Estimator of autocorrelation.

    """

    def __init__(self, frame_length, acr_order, norm=False, estimator="none"):
        super().__init__()

        assert 0 <= acr_order < frame_length

        self.frame_length = frame_length
        self.acr_order = acr_order
        self.norm = norm
        const = self._precompute(frame_length, acr_order, estimator)
        if torch.is_tensor(const):
            self.register_buffer("const", const)
        else:
            self.const = const

    def forward(self, x):
        """Estimate autocorrelation of input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Framed waveform.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            Autocorrelation.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> acorr = diffsptk.Autocorrelation(5, 3)
        >>> r = acorr(x)
        >>> r
        tensor([30.0000, 20.0000, 11.0000,  4.0000])

        """
        check_size(x.size(-1), self.frame_length, "length of waveform")
        return self._forward(x, self.acr_order, self.norm, self.const)

    @staticmethod
    def _forward(x, acr_order, norm, const):
        fft_length = x.size(-1) + acr_order
        if fft_length % 2 == 1:
            fft_length += 1
        X = torch.square(torch.fft.rfft(x, n=fft_length).abs())
        r = torch.fft.irfft(X)[..., : acr_order + 1] * const
        if norm:
            r = r / r[..., :1]
        return r

    @staticmethod
    def _func(x, acr_order, norm=False, estimator="none"):
        const = Autocorrelation._precompute(
            x.size(-1), acr_order, estimator, device=x.device
        )
        return Autocorrelation._forward(x, acr_order, norm, const)

    @staticmethod
    def _precompute(frame_length, acr_order, estimator, device=None):
        if estimator in (0, 1, "none"):
            return 1
        elif estimator in (2, "biased"):
            return 1 / frame_length
        elif estimator in (3, "unbiased"):
            return torch.arange(
                frame_length, frame_length - acr_order - 1, -1, device=device
            ).reciprocal()
        raise ValueError(f"estimator {estimator} is not supported.")
