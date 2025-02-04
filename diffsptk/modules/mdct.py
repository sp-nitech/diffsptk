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
import torch.nn.functional as F
from torch import nn

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

    def __init__(self, frame_length, window="sine", **kwargs):
        super().__init__()

        self.frame_period = frame_length // 2

        self.mdct = nn.Sequential(
            Frame(frame_length, self.frame_period),
            Window(frame_length, window=window, norm="none"),
            ModifiedDiscreteTransform(frame_length, window, **kwargs),
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
    def _func(x, frame_length, window, **kwargs):
        frame_period = frame_length // 2
        x = F.pad(x, (0, frame_period))
        y = Frame._func(x, frame_length, frame_period, True, False)
        y = Window._func(y, None, window, "none")
        y = ModifiedDiscreteTransform._func(y, window, **kwargs)
        return y


class ModifiedDiscreteTransform(nn.Module):
    """Oddly stacked modified discrete cosine/sine transform module.

    Parameters
    ----------
    length : int >= 2
        Input length, :math:`L`.

    window : bool or str
        If True, assume that input is windowed.

    transform : ['cosine', 'sine']
        Transform type.

    """

    def __init__(self, length, window, transform="cosine"):
        super().__init__()

        assert 2 <= length
        assert length % 2 == 0

        self.length = length
        self.register_buffer("W", self._precompute(length, window, transform))

    def forward(self, x):
        """Apply MDCT/MDST to input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Input.

        Returns
        -------
        out : Tensor [shape=(..., L/2)]
            Output.

        """
        check_size(x.size(-1), self.length, "dimension of input")
        return self._forward(x, self.W)

    @staticmethod
    def _forward(x, W):
        return torch.matmul(x, W)

    @staticmethod
    def _func(x, window, **kwargs):
        W = ModifiedDiscreteTransform._precompute(
            x.size(-1), window, dtype=x.dtype, device=x.device, **kwargs
        )
        return ModifiedDiscreteTransform._forward(x, W)

    @staticmethod
    def _precompute(length, window, transform="cosine", dtype=None, device=None):
        L2 = length
        L = L2 // 2
        n = torch.arange(L2, dtype=torch.double, device=device) + 0.5
        k = (torch.pi / L) * n[:L]
        n += L / 2

        z = 2 / L
        if window != "rectangular" or window is True:
            z *= 2
        z **= 0.5

        if transform == "cosine":
            W = z * torch.cos(k.unsqueeze(0) * n.unsqueeze(1))
        elif transform == "sine":
            W = z * torch.sin(k.unsqueeze(0) * n.unsqueeze(1))
        else:
            raise ValueError("transform must be either 'cosine' or 'sine'.")
        return to(W, dtype=dtype)
