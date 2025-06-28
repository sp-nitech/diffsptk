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
import math

import torch
from torch import nn

from ..typing import Callable, Precomputed
from ..utils.private import check_size, filter_values, get_layer
from .base import BaseFunctionalModule
from .mgc2mgc import MelGeneralizedCepstrumToMelGeneralizedCepstrum


class MelGeneralizedCepstrumToSpectrum(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mgc2sp.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the mel-cepstrum, :math:`M`.

    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    gamma : float in [-1, 1]
        The gamma parameter, :math:`\\gamma`.

    norm : bool
        If True, the input is assumed to be normalized.

    mul : bool
        If True, the input is assumed to be gamma-multiplied.

    n_fft : int >> L
        The number of FFT bins. Accurate conversion requires a large value.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power', \
                  'cycle', 'radian', 'degree', 'complex']
        The output format.

    """

    def __init__(
        self,
        cep_order: int,
        fft_length: int,
        *,
        alpha: float = 0,
        gamma: float = 0,
        norm: bool = False,
        mul: bool = False,
        n_fft=512,
        out_format: str | int = "power",
    ) -> None:
        super().__init__()

        self.in_dim = cep_order + 1

        self.values, layers, _ = self._precompute(**filter_values(locals()))
        self.layers = nn.ModuleList(layers)

    def forward(self, mc: torch.Tensor) -> torch.Tensor:
        """Convert mel-cepstrum to spectrum.

        Parameters
        ----------
        mc : Tensor [shape=(..., M+1)]
            Mel-cepstrum.

        Returns
        -------
        out : Tensor [shape=(..., L/2+1)]
            Spectrum.

        Examples
        --------
        >>> x = diffsptk.ramp(19)
        >>> stft = diffsptk.STFT(frame_length=10, frame_period=10, fft_length=16)
        >>> mcep = diffsptk.MelCepstralAnalysis(3, 16, 0.1, n_iter=1)
        >>> mc = mcep(stft(x))
        >>> mc
        tensor([[-0.8851,  0.7917, -0.1737,  0.0175],
                [-0.3522,  4.4222, -1.0882, -0.0511]])
        >>> mc2sp = diffsptk.MelGeneralizedCepstrumToSpectrum(3, 8, 0.1)
        >>> sp = mc2sp(mc)
        >>> sp
        tensor([[6.0634e-01, 4.6702e-01, 1.7489e-01, 4.4821e-02, 2.3869e-02],
                [3.5677e+02, 1.9435e+02, 6.0078e-01, 2.4278e-04, 8.8537e-06]])

        """
        check_size(mc.size(-1), self.in_dim, "dimension of cepstrum")
        return self._forward(mc, *self.values, *self.layers)

    @staticmethod
    def _func(mc: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, layers, _ = MelGeneralizedCepstrumToSpectrum._precompute(
            mc.size(-1) - 1, *args, **kwargs
        )
        return MelGeneralizedCepstrumToSpectrum._forward(mc, *values, *layers)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check() -> None:
        pass

    @staticmethod
    def _precompute(
        cep_order: int,
        fft_length: int,
        alpha: float,
        gamma: float,
        norm: bool,
        mul: bool,
        n_fft: int,
        out_format: str | int,
    ) -> Precomputed:
        MelGeneralizedCepstrumToSpectrum._check()

        if out_format in (0, "db"):
            formatter = lambda x: x.real * (20 / math.log(10))
        elif out_format in (1, "log-magnitude"):
            formatter = lambda x: x.real
        elif out_format in (2, "magnitude"):
            formatter = lambda x: torch.exp(x.real)
        elif out_format in (3, "power"):
            formatter = lambda x: torch.exp(2 * x.real)
        elif out_format in (4, "cycle"):
            formatter = lambda x: x.imag / torch.pi
        elif out_format in (5, "radian"):
            formatter = lambda x: x.imag
        elif out_format in (6, "degree"):
            formatter = lambda x: x.imag * (180 / torch.pi)
        elif out_format == "complex":
            formatter = lambda x: torch.polar(torch.exp(x.real), x.imag)
        else:
            raise ValueError(f"out_format {out_format} is not supported.")

        mgc2c = get_layer(
            inspect.stack()[1].function != "_func",
            MelGeneralizedCepstrumToMelGeneralizedCepstrum,
            dict(
                in_order=cep_order,
                in_alpha=alpha,
                in_gamma=gamma,
                in_norm=norm,
                in_mul=mul,
                out_order=fft_length // 2,
                out_alpha=0,
                out_gamma=0,
                out_norm=False,
                out_mul=False,
                n_fft=n_fft,
            ),
        )
        return (formatter,), (mgc2c,), None

    @staticmethod
    def _forward(
        mc: torch.Tensor,
        formatter: Callable,
        mgc2c: Callable,
    ) -> torch.Tensor:
        c = mgc2c(mc)
        sp = torch.fft.rfft(c, n=(c.size(-1) - 1) * 2)
        sp = formatter(sp)
        return sp
