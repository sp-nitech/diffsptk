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

from ..misc.utils import check_size
from .mdct import ModifiedDiscreteCosineTransformCore as MDCT
from .unframe import Unframe
from .window import Window


class InverseModifiedDiscreteCosineTransform(nn.Module):
    """This is the opposite module to :func:`~diffsptk.ModifiedDiscreteCosineTransform`.

    Parameters
    ----------
    frame_length : int >= 2
        Frame length, :math:`L`.

    window : ['sine', 'vorbis', 'kbd', 'rectangular']
        Window type.

    """

    def __init__(self, frame_length, window="sine"):
        super().__init__()

        self.frame_period = frame_length // 2

        self.imdct = InverseModifiedDiscreteCosineTransformCore(
            frame_length, window != "rectangular"
        )
        self.window = Window(frame_length, window=window, norm="none")
        self.unframe = Unframe(frame_length, self.frame_period)

    def forward(self, y, out_length=None):
        """Compute inverse modified discrete cosine transform.

        Parameters
        ----------
        y : Tensor [shape=(..., 2T/L, L/2)]
            Spectrum.

        out_length : int or None
            Length of output waveform.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            Reconstructed waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> x
        tensor([0., 1., 2., 3.])
        >>> mdct_params = {"frame_length": 4, "window": "vorbis"}
        >>> mdct = diffsptk.MDCT(**mdct_params)
        >>> imdct = diffsptk.IMDCT(**mdct_params)
        >>> y = imdct(mdct(x))
        >>> y
        tensor([1.0431e-07, 1.0000e+00, 2.0000e+00, 3.0000e+00])

        """
        x = self.imdct(y)
        x = self.window(x)
        x = self.unframe(x, out_length=out_length)
        if out_length is None:
            x = x[..., : -self.frame_period]
        return x

    @staticmethod
    def _func(y, out_length, frame_length, window):
        frame_period = frame_length // 2
        x = InverseModifiedDiscreteCosineTransformCore._func(y, window != "rectangular")
        x = Window._func(x, None, window=window, norm="none")
        x = Unframe._func(
            x,
            out_length,
            frame_length,
            frame_period,
            center=True,
            window="rectangular",
            norm="none",
        )
        if out_length is None:
            x = x[..., :-frame_period]
        return x


class InverseModifiedDiscreteCosineTransformCore(nn.Module):
    """Inverse modified discrete cosine transform module.

    Parameters
    ----------
    dct_length : int >= 2
        DCT length, :math:`L`.

    windowed : bool
        If True, assume that input is windowed.

    """

    def __init__(self, dct_length, windowed=False):
        super().__init__()

        assert 2 <= dct_length
        assert dct_length % 2 == 0

        self.dct_length = dct_length
        self.register_buffer("W", self._precompute(self.dct_length, windowed))

    def forward(self, y):
        """Apply inverse MDCT to input.

        Parameters
        ----------
        y : Tensor [shape=(..., L/2)]
            Input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            Inverse MDCT output.

        """
        check_size(2 * y.size(-1), self.dct_length, "dimension of input")
        return self._forward(y, self.W)

    @staticmethod
    def _forward(y, W):
        return torch.matmul(y, W)

    @staticmethod
    def _func(y, windowed):
        W = InverseModifiedDiscreteCosineTransformCore._precompute(
            2 * y.size(-1), windowed, dtype=y.dtype, device=y.device
        )
        return InverseModifiedDiscreteCosineTransformCore._forward(y, W)

    @staticmethod
    def _precompute(length, windowed, dtype=None, device=None):
        return MDCT._precompute(length, windowed, dtype=dtype, device=device).T
