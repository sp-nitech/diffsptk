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

    norm : bool
        If True, normalize the autocorrelation.

    out_format : ['none', 'biased', 'unbiased']
        Output format.

    """

    def __init__(self, frame_length, acr_order, norm=False, out_format="none"):
        super(Autocorrelation, self).__init__()

        assert 0 <= acr_order < frame_length

        self.frame_length = frame_length
        self.acr_order = acr_order
        self.norm = norm
        self.register_buffer(
            "const", self._precompute(frame_length, acr_order, out_format)
        )

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
        return self._forward(x, self.acr_order, self.norm, self.const)

    @staticmethod
    def _forward(x, acr_order, norm, const):
        fft_length = x.size(-1) + acr_order
        if fft_length % 2 == 1:
            fft_length += 1
        X = torch.square(torch.fft.rfft(x, n=fft_length).abs())
        r = torch.fft.irfft(X)[..., : acr_order + 1]
        r *= const
        if norm:
            r = r / r[..., :1]
        return r

    @staticmethod
    def _func(x, acr_order, norm=False, out_format="none"):
        const = Autocorrelation._precompute(
            x.size(-1), acr_order, out_format, dtype=x.dtype, device=x.device
        )
        return Autocorrelation._forward(x, acr_order, norm, const)

    @staticmethod
    def _precompute(frame_length, acr_order, out_format, dtype=None, device=None):
        if 0 <= out_format <= 1 or out_format == "none":
            return torch.tensor(1, dtype=dtype, device=device)
        elif out_format == 2 or out_format == "biased":
            return torch.full(
                (acr_order + 1,), 1 / frame_length, dtype=dtype, device=device
            )
        elif out_format == 3 or out_format == "unbiased":
            return torch.arange(
                frame_length, frame_length - acr_order - 1, -1, device=device
            ).reciprocal()
        raise ValueError(f"out_format {out_format} is not supported.")
