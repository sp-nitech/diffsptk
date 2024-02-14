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

    out_format : ['none', 'normalized', 'biased', 'unbiased']
        Output format.

    """

    def __init__(self, frame_length, acr_order, out_format="none"):
        super(Autocorrelation, self).__init__()

        assert 0 <= acr_order < frame_length

        self.frame_length = frame_length
        self.acr_order = acr_order
        self.out_format, const = self._precompute(frame_length, acr_order, out_format)
        self.register_buffer("const", const)

    def forward(self, x):
        """Estimate autocorrelation of input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Framed waveform.

        Returns
        -------
        Tensor [shape=(..., M+1)]
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
        return self._forward(x, self.acr_order, self.out_format, self.const)

    @staticmethod
    def _forward(x, acr_order, out_format, const):
        fft_length = x.size(-1) + acr_order
        if fft_length % 2 == 1:
            fft_length += 1
        X = torch.square(torch.fft.rfft(x, n=fft_length).abs())
        r = torch.fft.irfft(X)[..., : acr_order + 1]
        r = out_format(r, const)
        return r

    @staticmethod
    def _func(x, acr_order, out_format):
        const = Autocorrelation._precompute(
            x.size(-1), acr_order, out_format, dtype=x.dtype, device=x.device
        )
        return Autocorrelation._forward(x, acr_order, *const)

    @staticmethod
    def _precompute(frame_length, acr_order, out_format, dtype=None, device=None):
        if out_format == 0 or out_format == "none":
            return (
                lambda x, c: x,
                torch.tensor(1, dtype=dtype, device=device),
            )
        elif out_format == 1 or out_format == "normalized":
            return (
                lambda x, c: x / x[..., :1],
                torch.tensor(1, dtype=dtype, device=device),
            )
        elif out_format == 2 or out_format == "biased":
            return (
                lambda x, c: x * c,
                torch.tensor(1 / frame_length, dtype=dtype, device=device),
            )
        elif out_format == 3 or out_format == "unbiased":
            return (
                lambda x, c: x * c,
                torch.arange(
                    frame_length, frame_length - acr_order - 1, -1, device=device
                ).reciprocal(),
            )
        raise ValueError(f"out_format {out_format} is not supported.")
