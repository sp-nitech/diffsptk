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

import numpy as np
import torch
import torch.nn as nn

from ..misc.utils import numpy_to_torch
from .dct import DiscreteCosineTransform
from .fbank import MelFilterBankAnalysis


class MelFrequencyCepstralCoefficientsAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mfcc.html>`_
    for details.

    Parameters
    ----------
    mfcc_order : int >= 1 [scalar]
        Order of MFCC, :math:`M`.

    n_channel : int >= 1 [scalar]
        Number of mel-filter banks, :math:`C`.

    fft_length : int >= 2 [scalar]
        Number of FFT bins, :math:`L`.

    sample_rate : int >= 1 [scalar]
        Sample rate in Hz.

    lifter : int >= 1 [scalar]
        Liftering coefficient.

    f_min : float >= 0 [scalar]
        Minimum frequency in Hz.

    f_max : float <= sample_rate // 2 [scalar]
        Maximum frequency in Hz.

    floor : float > 0 [scalar]
        Minimum mel-filter bank output in linear scale.

    out_format : ['y', 'yE', 'yc', 'ycE']
        `y` is MFCC, `c` is C0, and `E` is energy.

    """

    def __init__(
        self,
        mfcc_order,
        n_channel,
        fft_length,
        sample_rate,
        lifter=1,
        out_format="y",
        **fbank_kwargs,
    ):
        super(MelFrequencyCepstralCoefficientsAnalysis, self).__init__()

        self.mfcc_order = mfcc_order

        assert 1 <= self.mfcc_order < n_channel

        if out_format == 0 or out_format == "y":
            self.format_func = lambda y, c, E: y
        elif out_format == 1 or out_format == "yE":
            self.format_func = lambda y, c, E: torch.cat((y, E), dim=-1)
        elif out_format == 2 or out_format == "yc":
            self.format_func = lambda y, c, E: torch.cat((y, c), dim=-1)
        elif out_format == 3 or out_format == "ycE":
            self.format_func = lambda y, c, E: torch.cat((y, c, E), dim=-1)
        else:
            raise ValueError(f"out_format {out_format} is not supported")

        self.fbank = MelFilterBankAnalysis(
            n_channel, fft_length, sample_rate, out_format="y,E", **fbank_kwargs
        )
        self.dct = DiscreteCosineTransform(n_channel)

        m = np.arange(1, self.mfcc_order + 1)
        liftering_vector = 1 + (lifter / 2) * np.sin((np.pi / lifter) * m)
        self.register_buffer("liftering_vector", numpy_to_torch(liftering_vector))

        self.const = np.sqrt(2)

    def forward(self, x):
        """Compute MFCC.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            Power spectrum.

        Returns
        -------
        y : Tensor [shape=(..., M)]
            MFCC without C0.

        E : Tensor [shape=(..., 1)]
            Energy.

        c : Tensor [shape=(..., 1)]
            C0.

        Examples
        --------
        >>> x = diffsptk.ramp(19)
        >>> stft = diffsptk.STFT(frame_length=10, frame_period=10, fft_length=32)
        >>> mfcc = diffsptk.MFCC(4, 8, 32, 8000)
        >>> y = mfcc(stft(x))
        >>> y
        tensor([[-7.7745e-03, -1.4447e-02,  1.6157e-02,  1.1069e-03],
                [ 2.8049e+00, -1.6257e+00, -2.3566e-02,  1.2804e-01]])

        """
        y, E = self.fbank(x)
        y = self.dct(y)
        c = y[..., :1] * self.const
        y = y[..., 1 : self.mfcc_order + 1] * self.liftering_vector
        return self.format_func(y, c, E)
