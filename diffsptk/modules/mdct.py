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
import torch.nn.functional as F

from ..misc.utils import check_size
from ..misc.utils import to
from .frame import Frame
from .window import Window


class ModifiedDiscreteCosineTransform(nn.Module):
    """This module is a simple cascade of framing, windowing, and modified DCT.

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

        self.mdct = nn.Sequential(
            Frame(frame_length, self.frame_period),
            Window(frame_length, window=window, norm="none"),
            ModifiedDiscreteCosineTransformCore(frame_length, window != "rectangular"),
        )

    def forward(self, x):
        """Compute modified discrete cosine transform.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Waveform.

        Returns
        -------
        out : Tensor [shape=(..., 2T/L, L/2)]
            Spectrum.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> x
        tensor([0., 1., 2., 3.])
        >>> mdct = diffsptk.MDCT(frame_length=4)
        >>> y = mdct(x)
        >>> y
        tensor([[-0.3536, -0.1464],
                [-3.1213, -0.2929],
                [-0.7678,  1.8536]])

        """
        # This is for perfect reconstruction.
        x = F.pad(x, (0, self.frame_period))
        return self.mdct(x)

    @staticmethod
    def _func(x, frame_length, window):
        frame_period = frame_length // 2
        x = F.pad(x, (0, frame_period))
        y = Frame._func(x, frame_length, frame_period, True, False)
        y = Window._func(y, None, window, "none")
        y = ModifiedDiscreteCosineTransformCore._func(y, window != "rectangular")
        return y


class ModifiedDiscreteCosineTransformCore(nn.Module):
    """Modified discrete cosine transform module.

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

    def forward(self, x):
        """Apply MDCT to input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Input.

        Returns
        -------
        out : Tensor [shape=(..., L/2)]
            MDCT output.

        """
        check_size(x.size(-1), self.dct_length, "dimension of input")
        return self._forward(x, self.W)

    @staticmethod
    def _forward(x, W):
        return torch.matmul(x, W)

    @staticmethod
    def _func(x, windowed):
        W = ModifiedDiscreteCosineTransformCore._precompute(
            x.size(-1), windowed, dtype=x.dtype, device=x.device
        )
        return ModifiedDiscreteCosineTransformCore._forward(x, W)

    @staticmethod
    def _precompute(length, windowed, dtype=None, device=None):
        L2 = length
        L = L2 // 2
        n = torch.arange(L2, dtype=torch.double, device=device) + 0.5
        k = (torch.pi / L) * n[:L]
        n += L / 2
        z = (4 / L) ** 0.5 if windowed else (2 / L) ** 0.5
        W = z * torch.cos(k.unsqueeze(0) * n.unsqueeze(1))
        return to(W, dtype=dtype)
