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

from ..utils.private import check_size
from ..utils.private import get_layer
from ..utils.private import get_values
from ..utils.private import to
from .base import BaseFunctionalModule
from .frame import Frame
from .window import Window


class ModifiedDiscreteCosineTransform(BaseFunctionalModule):
    """This module is a simple cascade of framing, windowing, and modified DCT.

    Parameters
    ----------
    frame_length : int >= 2
        The frame length, :math:`L`.

    window : ['sine', 'vorbis', 'kbd', 'rectangular']
        The window type.

    """

    def __init__(self, frame_length, window="sine"):
        super().__init__()

        self.values, layers, _ = self._precompute(*get_values(locals()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """Compute modified discrete cosine transform.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(..., 2T/L, L/2)]
            The spectrum.

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
        return self._forward(x, *self.values, *self.layers)

    @staticmethod
    def _func(x, *args, **kwargs):
        values, layers, _ = ModifiedDiscreteCosineTransform._precompute(
            *args, **kwargs, module=False
        )
        return ModifiedDiscreteCosineTransform._forward(x, *values, *layers)

    @staticmethod
    def _takes_input_size():
        return False

    @staticmethod
    def _check():
        pass

    @staticmethod
    def _precompute(frame_length, window, transform="cosine", module=True):
        ModifiedDiscreteCosineTransform._check()
        frame_period = frame_length // 2

        frame = get_layer(
            module,
            Frame,
            dict(
                frame_length=frame_length,
                frame_period=frame_period,
            ),
        )
        window_ = get_layer(
            module,
            Window,
            dict(
                in_length=frame_length,
                out_length=None,
                window=window,
                norm="none",
            ),
        )
        mdt = get_layer(
            module,
            ModifiedDiscreteTransform,
            dict(
                length=frame_length,
                window=window,
                transform=transform,
            ),
        )
        return (frame_period,), (frame, window_, mdt), None

    @staticmethod
    def _forward(x, frame_period, frame, window, mdt):
        # This padding is for perfect reconstruction.
        x = F.pad(x, (0, frame_period))
        return mdt(window(frame(x)))


class ModifiedDiscreteTransform(BaseFunctionalModule):
    """Oddly stacked modified discrete cosine/sine transform module.

    Parameters
    ----------
    length : int >= 2
        The input length, :math:`L`.

    window : str
        The window type used to determine whether it is rectangular or not.

    transform : ['cosine', 'sine']
        The transform type.

    """

    def __init__(self, length, window, transform="cosine"):
        super().__init__()

        self.in_dim = length

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("W", tensors[0])

    def forward(self, x):
        """Apply MDCT/MDST to the input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            The input.

        Returns
        -------
        out : Tensor [shape=(..., L/2)]
            The output.

        """
        check_size(x.size(-1), self.in_dim, "dimension of input")
        return self._forward(x, **self._buffers)

    @staticmethod
    def _func(x, *args, **kwargs):
        _, _, tensors = ModifiedDiscreteTransform._precompute(
            x.size(-1), *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return ModifiedDiscreteTransform._forward(x, *tensors)

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(length):
        if length < 2 or length % 2 == 1:
            raise ValueError("length must be at least 2 and even.")

    @staticmethod
    def _precompute(length, window, transform="cosine", device=None, dtype=None):
        ModifiedDiscreteTransform._check(length)
        L2 = length
        L = L2 // 2
        n = torch.arange(L2, device=device, dtype=torch.double) + 0.5
        k = (torch.pi / L) * n[:L]
        n += L / 2

        z = 2 / L
        if window != "rectangular":
            z *= 2
        z **= 0.5

        if transform == "cosine":
            W = z * torch.cos(k.unsqueeze(0) * n.unsqueeze(1))
        elif transform == "sine":
            W = z * torch.sin(k.unsqueeze(0) * n.unsqueeze(1))
        else:
            raise ValueError("transform must be either 'cosine' or 'sine'.")
        return None, None, (to(W, dtype=dtype),)

    @staticmethod
    def _forward(x, W):
        return torch.matmul(x, W)
