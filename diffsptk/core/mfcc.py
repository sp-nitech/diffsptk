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

from ..misc.utils import default_dtype
from ..misc.utils import is_in
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
        Floor value of raw filter bank output.

    out_format : ['y', 'yE', 'yc', 'y,E', 'y,c']
        `y` is MFCC, `c` is C0, and `E` is energy. If this is `y?`, the two output
        tensors are concatenated and return the tensor instead of the tuple.

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
        self.out_format = out_format

        assert 1 <= self.mfcc_order and self.mfcc_order < n_channel
        assert is_in(self.out_format, ["y", "yE", "yc", "y,E", "y,c"])

        self.fbank = MelFilterBankAnalysis(
            n_channel, fft_length, sample_rate, out_format="y,E", **fbank_kwargs
        )
        self.dct = DiscreteCosineTransform(n_channel)

        m = np.arange(1, mfcc_order + 1, dtype=default_dtype())
        liftering_vector = 1 + (lifter / 2) * np.sin((np.pi / lifter) * m)
        self.register_buffer("liftering_vector", torch.from_numpy(liftering_vector))

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
        c = y[..., :1] * np.sqrt(2)
        y = y[..., 1 : self.mfcc_order + 1] * self.liftering_vector

        if self.out_format == "y":
            return y
        elif self.out_format == "yE":
            return torch.cat((y, E), dim=-1)
        elif self.out_format == "yc":
            return torch.cat((y, c), dim=-1)
        elif self.out_format == "y,E":
            return y, E
        elif self.out_format == "y,c":
            return y, c
        else:
            raise RuntimeError
