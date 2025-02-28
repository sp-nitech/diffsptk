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

from ..misc.utils import Lambda
from ..misc.utils import get_layer
from ..misc.utils import get_values
from .base import BaseFunctionalModule
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

    """

    def __init__(
        self,
        frame_length,
        frame_period,
        fft_length,
        *,
        center=True,
        window="blackman",
        norm="power",
    ):
        super().__init__()

        _, layers, _ = self._precompute(*get_values(locals()))
        self.layers = nn.ModuleList(layers)

    def forward(self, y, out_length=None):
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
    def _func(x, out_length, *args, **kwargs):
        _, layers, _ = InverseShortTimeFourierTransform._precompute(
            *args,
            **kwargs,
            device=x.device,
            dtype=torch.float if x.dtype == torch.complex64 else torch.double,
        )
        return InverseShortTimeFourierTransform._forward(x, out_length, *layers)

    @staticmethod
    def _takes_input_size():
        return False

    @staticmethod
    def _check():
        pass

    @staticmethod
    def _precompute(
        frame_length,
        frame_period,
        fft_length,
        center,
        window,
        norm,
        device=None,
        dtype=None,
    ):
        InverseShortTimeFourierTransform._check()
        module = inspect.stack()[1].function == "__init__"

        ifft = Lambda(lambda x: torch.fft.irfft(x, n=fft_length)[..., :frame_length])
        unframe = get_layer(
            module,
            Unframe,
            dict(
                frame_length=frame_length,
                frame_period=frame_period,
                center=center,
                norm=norm,
                window=window,
            ),
        )
        return None, (ifft, unframe), None

    @staticmethod
    def _forward(y, out_length, ifft, unframe):
        return unframe(ifft(y), out_length)
