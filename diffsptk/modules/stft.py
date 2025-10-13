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
from ..utils.private import filter_values, get_layer
from .base import BaseFunctionalModule
from .fftr import RealValuedFastFourierTransform
from .frame import Frame
from .spec import Spectrum
from .window import Window

LEARNABLES = ("basis", "window")


class ShortTimeFourierTransform(BaseFunctionalModule):
    """This module is a simple cascade of framing, windowing, and spectrum calculation.

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

    zmean : bool
        If True, perform mean subtraction on each frame.

    mode : ['constant', 'reflect', 'replicate', 'circular']
        The padding method.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular', 'nuttall']
        The window type.

    norm : ['none', 'power', 'magnitude']
        The normalization type of the window.

    symmetric : bool
        If True, the window is symmetric, otherwise periodic.

    eps : float >= 0
        A small value added to the power spectrum.

    relative_floor : float < 0 or None
        The relative floor of the power spectrum in dB.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power', 'complex']
        The output format.

    learnable : bool or list[str]
        Indicates whether the parameters are learnable. If a boolean, it specifies
        whether all parameters are learnable. If a list, it contains the keys of the
        learnable parameters, which can only be "basis" and "window".

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        frame_length: int,
        frame_period: int,
        fft_length: int,
        *,
        center: bool = True,
        zmean: bool = False,
        mode: str = "constant",
        window: str = "blackman",
        norm: str = "power",
        symmetric: bool = True,
        eps: float = 1e-9,
        relative_floor: float | None = None,
        out_format: str = "power",
        learnable: bool | list[str] = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        _, layers, _ = self._precompute(**filter_values(locals()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute short-time Fourier transform.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(..., T/P, N/2+1)]
            The output spectrogram.

        Examples
        --------
        >>> import diffsptk
        >>> stft = diffsptk.STFT(frame_length=3, frame_period=1, fft_length=8)
        >>> x = diffsptk.ramp(1, 3)
        >>> x
        tensor([1., 2., 3.])
        >>> y = stft(x)
        >>> y
        tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                [4.0000, 4.0000, 4.0000, 4.0000, 4.0000],
                [9.0000, 9.0000, 9.0000, 9.0000, 9.0000]])

        """
        return self._forward(x, *self.layers)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, layers, _ = ShortTimeFourierTransform._precompute(
            *args, **kwargs, learnable=False, device=x.device, dtype=x.dtype
        )
        return ShortTimeFourierTransform._forward(x, *layers)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(learnable: bool | list[str]) -> None:
        if isinstance(learnable, (tuple, list)):
            if any(x not in LEARNABLES for x in learnable):
                raise ValueError("An unsupported key is found in learnable.")
        elif not isinstance(learnable, bool):
            raise ValueError("learnable must be boolean or list.")

    @staticmethod
    def _precompute(
        frame_length: int,
        frame_period: int,
        fft_length: int,
        center: bool,
        zmean: bool,
        mode: str,
        window: str,
        norm: str,
        symmetric: bool,
        eps: float,
        relative_floor: float | None,
        out_format: str,
        learnable: bool | list[str],
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        ShortTimeFourierTransform._check(learnable)
        module = inspect.stack()[1].function != "_func"

        if learnable is True:
            learnable = LEARNABLES
        elif learnable is False:
            learnable = ()

        frame = get_layer(
            module,
            Frame,
            dict(
                frame_length=frame_length,
                frame_period=frame_period,
                center=center,
                zmean=zmean,
                mode=mode,
            ),
        )
        window_ = get_layer(
            module,
            Window,
            dict(
                in_length=frame_length,
                out_length=fft_length,
                window=window,
                norm=norm,
                symmetric=symmetric,
                learnable="window" in learnable,
                device=device,
                dtype=dtype,
            ),
        )
        if out_format == "complex":
            spec = get_layer(
                module,
                RealValuedFastFourierTransform,
                dict(
                    fft_length=fft_length,
                    out_format="complex",
                    learnable="basis" in learnable,
                    device=device,
                    dtype=dtype,
                ),
            )
        else:
            spec = get_layer(
                module,
                Spectrum,
                dict(
                    fft_length=fft_length,
                    eps=eps,
                    relative_floor=relative_floor,
                    out_format=out_format,
                    learnable="basis" in learnable,
                ),
            )
        return None, (frame, window_, spec), None

    @staticmethod
    def _forward(
        x: torch.Tensor, frame: Callable, window: Callable, spec: Callable
    ) -> torch.Tensor:
        return spec(window(frame(x)))
