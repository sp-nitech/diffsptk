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

from ..typing import Callable, Precomputed
from ..utils.private import get_layer, get_values
from .base import BaseFunctionalModule
from .ifftr import RealValuedInverseFastFourierTransform
from .stft import LEARNABLES, ShortTimeFourierTransform
from .unframe import Unframe


class InverseShortTimeFourierTransform(BaseFunctionalModule):
    """This is the opposite module to :func:`~diffsptk.ShortTimeFourierTransform`.

    Parameters
    ----------
    frame_length : int >= 1
        The frame length in samples, :math:`L`.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    fft_length : int >= L
        The number of FFT bins, :math:`N`.

    center : bool
        If True, pad the input on both sides so that the frame is centered.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular', 'nuttall']
        The window type.

    norm : ['none', 'power', 'magnitude']
        The normalization type of the window.

    symmetric : bool
        If True, the window is symmetric, otherwise periodic.

    learnable : bool or list[str]
        Indicates whether the parameters are learnable. If a boolean, it specifies
        whether all parameters are learnable. If a list, it contains the keys of the
        learnable parameters, which can only be "basis" and "window".

    """

    def __init__(
        self,
        frame_length: int,
        frame_period: int,
        fft_length: int,
        *,
        center: bool = True,
        window: str = "blackman",
        norm: str = "power",
        symmetric: bool = True,
        learnable: bool | list[str] = False,
    ) -> None:
        super().__init__()

        _, layers, _ = self._precompute(*get_values(locals()))
        self.layers = nn.ModuleList(layers)

    def forward(self, y: torch.Tensor, out_length: int | None = None) -> torch.Tensor:
        """Compute inverse short-time Fourier transform.

        Parameters
        ----------
        y : Tensor [shape=(..., T/P, N/2+1)]
            The complex spectrogram.

        out_length : int > 0 or None
            The length of the output waveform.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The reconstructed waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 3)
        >>> x
        tensor([1., 2., 3.])
        >>> stft_params = {"frame_length": 3, "frame_period": 1, "fft_length": 8}
        >>> stft = diffsptk.STFT(**stft_params, out_format="complex")
        >>> istft = diffsptk.ISTFT(**stft_params)
        >>> y = istft(stft(x), out_length=3)
        >>> y
        tensor([1., 2., 3.])

        """
        return self._forward(y, out_length, *self.layers)

    @staticmethod
    def _func(x: torch.Tensor, out_length: int | None, *args, **kwargs) -> torch.Tensor:
        _, layers, _ = InverseShortTimeFourierTransform._precompute(*args, **kwargs)
        return InverseShortTimeFourierTransform._forward(x, out_length, *layers)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(*args, **kwargs) -> None:
        ShortTimeFourierTransform._check(*args, **kwargs)

    @staticmethod
    def _precompute(
        frame_length: int,
        frame_period: int,
        fft_length: int,
        center: bool,
        window: str,
        norm: str,
        symmetric: bool,
        learnable: bool | list[str] = False,
    ) -> Precomputed:
        InverseShortTimeFourierTransform._check(learnable)
        module = inspect.stack()[1].function == "__init__"

        if learnable is True:
            learnable = LEARNABLES
        elif learnable is False:
            learnable = ()

        ifftr = get_layer(
            module,
            RealValuedInverseFastFourierTransform,
            dict(
                fft_length=fft_length,
                out_length=frame_length,
                learnable="basis" in learnable,
            ),
        )
        unframe = get_layer(
            module,
            Unframe,
            dict(
                frame_length=frame_length,
                frame_period=frame_period,
                center=center,
                window=window,
                norm=norm,
                symmetric=symmetric,
                learnable="window" in learnable,
            ),
        )
        return None, (ifftr, unframe), None

    @staticmethod
    def _forward(
        y: torch.Tensor,
        out_length: int | None,
        ifftr: Callable,
        unframe: Callable,
    ) -> torch.Tensor:
        return unframe(ifftr(y), out_length)
