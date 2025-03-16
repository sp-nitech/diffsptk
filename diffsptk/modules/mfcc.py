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

import inspect

import torch
from torch import nn

from ..utils.private import get_layer
from ..utils.private import get_values
from ..utils.private import to
from .base import BaseFunctionalModule
from .dct import DiscreteCosineTransform
from .fbank import MelFilterBankAnalysis


class MelFrequencyCepstralCoefficientsAnalysis(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mfcc.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    mfcc_order : int >= 1
        The order of the MFCC, :math:`M`.

    n_channel : int >= 1
        The number of mel filter banks, :math:`C`.

    sample_rate : int >= 1
        The sample rate in Hz.

    lifter : int >= 1
        The liftering coefficient.

    f_min : float >= 0
        The minimum frequency in Hz.

    f_max : float <= sample_rate // 2
        The maximum frequency in Hz.

    floor : float > 0
        The minimum mel filter bank output in linear scale.

    out_format : ['y', 'yE', 'yc', 'ycE']
        `y` is MFCC, `c` is C0, and `E` is energy.

    References
    ----------
    .. [1] S. Young et al., "The HTK Book," *Cambridge University Press*, 2006.

    """

    def __init__(
        self,
        *,
        fft_length,
        mfcc_order,
        n_channel,
        sample_rate,
        lifter=1,
        f_min=0,
        f_max=None,
        floor=1e-5,
        out_format="y",
    ):
        super().__init__()

        self.values, layers, tensors = self._precompute(*get_values(locals()))
        self.layers = nn.ModuleList(layers)
        self.register_buffer("liftering_vector", tensors[0])

    def forward(self, x):
        """Compute the MFCC from the power spectrum.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            The power spectrum.

        Returns
        -------
        y : Tensor [shape=(..., M)]
            The MFCC without C0.

        E : Tensor [shape=(..., 1)] (optional)
            The energy.

        c : Tensor [shape=(..., 1)] (optional)
            The C0.

        Examples
        --------
        >>> x = diffsptk.ramp(19)
        >>> stft = diffsptk.STFT(frame_length=10, frame_period=10, fft_length=32)
        >>> mfcc = diffsptk.MFCC(
        ...     fft_length=32, mfcc_order=4, n_channel=8, sample_rate=8000
        ... )
        >>> y = mfcc(stft(x))
        >>> y
        tensor([[-7.7745e-03, -1.4447e-02,  1.6157e-02,  1.1069e-03],
                [ 2.8049e+00, -1.6257e+00, -2.3566e-02,  1.2804e-01]])

        """
        return self._forward(x, *self.values, *self.layers, **self._buffers)

    @staticmethod
    def _func(x, *args, **kwargs):
        values, layers, tensors = MelFrequencyCepstralCoefficientsAnalysis._precompute(
            2 * x.size(-1) - 2, *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return MelFrequencyCepstralCoefficientsAnalysis._forward(
            x, *values, *layers, *tensors
        )

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(mfcc_order, n_channel, lifter):
        if mfcc_order < 0:
            raise ValueError("mfcc_order must be non-negative.")
        if n_channel <= mfcc_order:
            raise ValueError("mfcc_order must be less than n_channel.")
        if lifter < 0:
            raise ValueError("lifter must be non-negative.")

    @staticmethod
    def _precompute(
        fft_length,
        mfcc_order,
        n_channel,
        sample_rate,
        lifter,
        f_min,
        f_max,
        floor,
        out_format,
        device=None,
        dtype=None,
    ):
        MelFrequencyCepstralCoefficientsAnalysis._check(mfcc_order, n_channel, lifter)
        module = inspect.stack()[1].function == "__init__"

        if out_format in (0, "y"):
            formatter = lambda y, c, E: y
        elif out_format in (1, "yE"):
            formatter = lambda y, c, E: torch.cat((y, E), dim=-1)
        elif out_format in (2, "yc"):
            formatter = lambda y, c, E: torch.cat((y, c), dim=-1)
        elif out_format in (3, "ycE"):
            formatter = lambda y, c, E: torch.cat((y, c, E), dim=-1)
        else:
            raise ValueError(f"out_format {out_format} is not supported.")

        fbank = get_layer(
            module,
            MelFilterBankAnalysis,
            dict(
                fft_length=fft_length,
                n_channel=n_channel,
                sample_rate=sample_rate,
                f_min=f_min,
                f_max=f_max,
                floor=floor,
                use_power=False,
                out_format="y,E",
            ),
        )
        dct = get_layer(
            module,
            DiscreteCosineTransform,
            dict(
                dct_length=n_channel,
                dct_type=2,
            ),
        )

        ramp = torch.arange(mfcc_order + 1, device=device, dtype=torch.double)
        liftering_vector = 1 + (lifter / 2) * torch.sin((torch.pi / lifter) * ramp)
        liftering_vector[0] = 2**0.5

        return (formatter,), (fbank, dct), (to(liftering_vector, dtype=dtype),)

    @staticmethod
    def _forward(x, formatter, fbank, dct, liftering_vector):
        y, E = fbank(x)
        y = dct(y)
        y = y[..., : len(liftering_vector)] * liftering_vector
        c, y = torch.split(y, [1, y.size(-1) - 1], dim=-1)
        return formatter(y, c, E)
