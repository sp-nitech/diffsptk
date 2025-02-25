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

from ..misc.utils import get_values
from ..misc.utils import numpy_to_torch
from ..misc.utils import replicate1
from ..misc.utils import to
from .base import BaseFunctionalModule
from .fbank import MelFilterBankAnalysis
from .levdur import LevinsonDurbin
from .mgc2mgc import MelGeneralizedCepstrumToMelGeneralizedCepstrum


class PerceptualLinearPredictiveCoefficientsAnalysis(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/plp.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    plp_order : int >= 1
        The order of the PLP, :math:`M`.

    n_channel : int >= 1
        Number of mel filter banks, :math:`C`.

    sample_rate : int >= 1
        The sample rate in Hz.

    compression_factor : float > 0
        The amplitude compression factor.

    lifter : int >= 1
        The liftering coefficient.

    f_min : float >= 0
        The minimum frequency in Hz.

    f_max : float <= sample_rate // 2
        The maximum frequency in Hz.

    floor : float > 0
        The minimum mel filter bank output in linear scale.

    n_fft : int >> M
        The number of FFT bins for the conversion from LPC to cepstrum.
        The accurate conversion requires the large value.

    out_format : ['y', 'yE', 'yc', 'ycE']
        `y` is PLP, `c` is C0, and `E` is energy.

    References
    ----------
    .. [1] S. Young et al., "The HTK Book," *Cambridge University Press*, 2006.

    """

    def __init__(
        self,
        *,
        fft_length,
        plp_order,
        n_channel,
        sample_rate,
        compression_factor=0.33,
        lifter=1,
        f_min=0,
        f_max=None,
        floor=1e-5,
        n_fft=512,
        out_format="y",
    ):
        super().__init__()

        self.values, layers, tensors = self._precompute(*get_values(locals()))
        self.layers = nn.ModuleList(layers)
        self.register_buffer("equal_loudness_curve", tensors[0])
        self.register_buffer("liftering_vector", tensors[1])

    def forward(self, x):
        """Compute the PLP from the power spectrum.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            The power spectrum.

        Returns
        -------
        y : Tensor [shape=(..., M)]
            The PLP without C0.

        E : Tensor [shape=(..., 1)] (optional)
            The energy.

        c : Tensor [shape=(..., 1)] (optional)
            The C0.

        Examples
        --------
        >>> x = diffsptk.ramp(19)
        >>> stft = diffsptk.STFT(frame_length=10, frame_period=10, fft_length=32)
        >>> plp = diffsptk.PLP(4, 8, 32, 8000)
        >>> y = plp(stft(x))
        >>> y
        tensor([[-0.2896, -0.2356, -0.0586, -0.0387],
                [ 0.4468, -0.5820,  0.0104, -0.0505]])

        """
        return self._forward(x, *self.values, *self.layers, **self._buffers)

    @staticmethod
    def _func(x, *args, **kwargs):
        values, layers, tensors = (
            PerceptualLinearPredictiveCoefficientsAnalysis._precompute(
                2 * x.size(-1) - 2, *args, **kwargs, device=x.device, dtype=x.dtype
            )
        )
        return PerceptualLinearPredictiveCoefficientsAnalysis._forward(
            x, *values, *layers, *tensors
        )

    @staticmethod
    def _check(plp_order, n_channel, compression_factor, lifter):
        if plp_order < 0:
            raise ValueError("plp_order must be non-negative.")
        if n_channel <= plp_order:
            raise ValueError("plp_order must be less than n_channel.")
        if compression_factor <= 0:
            raise ValueError("compression_factor must be positive.")
        if lifter < 0:
            raise ValueError("lifter must be non-negative.")

    @staticmethod
    def _precompute(
        fft_length,
        plp_order,
        n_channel,
        sample_rate,
        compression_factor,
        lifter,
        f_min,
        f_max,
        floor,
        n_fft,
        out_format,
        device=None,
        dtype=None,
    ):
        PerceptualLinearPredictiveCoefficientsAnalysis._check(
            plp_order, n_channel, compression_factor, lifter
        )

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

        fbank = MelFilterBankAnalysis(
            fft_length=fft_length,
            n_channel=n_channel,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            floor=floor,
            use_power=True,
            out_format="y,E",
            device=device,
            dtype=dtype,
        )
        levdur = LevinsonDurbin(
            plp_order,
            device=device,
            dtype=dtype,
        )
        lpc2c = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
            plp_order,
            plp_order,
            in_gamma=-1,
            in_norm=True,
            in_mul=True,
            n_fft=n_fft,
        )

        f = fbank.center_frequencies[:-1] ** 2
        e = (f / (f + 1.6e5)) ** 2 * (f + 1.44e6) / (f + 9.61e6)
        equal_loudness_curve = numpy_to_torch(e)

        ramp = torch.arange(plp_order + 1, device=device, dtype=torch.double)
        liftering_vector = 1 + (lifter / 2) * torch.sin((torch.pi / lifter) * ramp)
        liftering_vector[0] = 2

        return (
            (compression_factor, formatter),
            (fbank, levdur, lpc2c),
            (
                to(equal_loudness_curve, device=device, dtype=dtype),
                to(liftering_vector, dtype=dtype),
            ),
        )

    @staticmethod
    def _forward(
        x,
        compression_factor,
        formatter,
        fbank,
        levdur,
        lpc2c,
        equal_loudness_curve,
        liftering_vector,
    ):
        y, E = fbank(x)
        y = (torch.exp(y) * equal_loudness_curve) ** compression_factor
        y = replicate1(y)
        y = torch.fft.hfft(y, norm="forward").real[..., : len(liftering_vector)]
        y = lpc2c(levdur(y)) * liftering_vector
        c, y = torch.split(y, [1, y.size(-1) - 1], dim=-1)
        return formatter(y, c, E)
