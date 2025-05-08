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

from ..typing import Callable, Precomputed
from ..utils.private import check_size, get_layer, get_values
from .base import BaseFunctionalModule
from .mdct import LEARNABLES, ModifiedDiscreteCosineTransform, ModifiedDiscreteTransform
from .unframe import Unframe
from .window import Window


class InverseModifiedDiscreteCosineTransform(BaseFunctionalModule):
    """This is the opposite module to :func:`~diffsptk.ModifiedDiscreteCosineTransform`.

    Parameters
    ----------
    frame_length : int >= 2
        The frame length, :math:`L`.

    window : ['sine', 'vorbis', 'kbd', 'rectangular']
        The window type.

    learnable : bool or list[str]
        Indicates whether the parameters are learnable. If a boolean, it specifies
        whether all parameters are learnable. If a list, it contains the keys of the
        learnable parameters, which can only be "basis" and "window".

    """

    def __init__(
        self,
        frame_length: int,
        window: str = "sine",
        learnable: bool | list[str] = False,
    ) -> None:
        super().__init__()

        self.values, layers, _ = self._precompute(*get_values(locals()))
        self.layers = nn.ModuleList(layers)

    def forward(self, y: torch.Tensor, out_length: int | None = None) -> torch.Tensor:
        """Compute inverse modified discrete cosine transform.

        Parameters
        ----------
        y : Tensor [shape=(..., 2T/L, L/2)]
            The spectrum.

        out_length : int or None
            The length of the output waveform.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The reconstructed waveform.

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
        return self._forward(y, out_length, *self.values, *self.layers)

    @staticmethod
    def _func(y: torch.Tensor, out_length: int | None, *args, **kwargs) -> torch.Tensor:
        values, layers, _ = InverseModifiedDiscreteCosineTransform._precompute(
            *args, **kwargs, module=False
        )
        return InverseModifiedDiscreteCosineTransform._forward(
            y, out_length, *values, *layers
        )

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(*args, **kwargs) -> None:
        ModifiedDiscreteCosineTransform._check(*args, **kwargs)

    @staticmethod
    def _precompute(
        frame_length: int,
        window: str,
        learnable: bool | list[str] = False,
        transform: str = "cosine",
        module: bool = True,
    ) -> Precomputed:
        InverseModifiedDiscreteCosineTransform._check(learnable)
        frame_period = frame_length // 2

        if learnable is True:
            learnable = LEARNABLES
        elif learnable is False:
            learnable = ()

        imdt = get_layer(
            module,
            InverseModifiedDiscreteTransform,
            dict(
                length=frame_length,
                window=window,
                transform=transform,
                learnable="basis" in learnable,
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
                symmetric=True,
                learnable="window" in learnable,
            ),
        )
        unframe = get_layer(
            module,
            Unframe,
            dict(
                frame_length=frame_length,
                frame_period=frame_period,
            ),
        )
        return (frame_period,), (imdt, window_, unframe), None

    @staticmethod
    def _forward(
        y: torch.Tensor,
        out_length: int | None,
        frame_period: int,
        imdt: Callable,
        window: Callable,
        unframe: Callable,
    ) -> torch.Tensor:
        x = unframe(window(imdt(y)), out_length=out_length)
        if out_length is None:
            x = x[..., :-frame_period]
        return x


class InverseModifiedDiscreteTransform(BaseFunctionalModule):
    """Oddly stacked inverse modified discrete cosine/sine transform module.

    Parameters
    ----------
    length : int >= 2
        The output length, :math:`L`.

    window : str
        The window type used to determine whether it is rectangular or not.

    transform : ['cosine', 'sine']
        The transform type.

    learnable : bool
        Whether to make the DCT matrix learnable.

    """

    def __init__(
        self,
        length: int,
        window: str,
        transform: str = "cosine",
        learnable: bool = False,
    ) -> None:
        super().__init__()

        self.in_dim = length // 2

        _, _, tensors = self._precompute(*get_values(locals(), drop=1))
        if learnable:
            self.W = nn.Parameter(tensors[0])
        else:
            self.register_buffer("W", tensors[0])

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse MDCT/MDST to the input.

        Parameters
        ----------
        y : Tensor [shape=(..., L/2)]
            The input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            The output.

        """
        check_size(y.size(-1), self.in_dim, "dimension of input")
        return self._forward(y, **self._buffers, **self._parameters)

    @staticmethod
    def _func(y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _, tensors = InverseModifiedDiscreteTransform._precompute(
            2 * y.size(-1),
            *args,
            **kwargs,
            device=y.device,
            dtype=y.dtype,
        )
        return InverseModifiedDiscreteTransform._forward(y, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(*args, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def _precompute(*args, **kwargs) -> Precomputed:
        _, _, tensors = ModifiedDiscreteTransform._precompute(*args, **kwargs)
        return None, None, (tensors[0].T,)

    @staticmethod
    def _forward(y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.matmul(y, W)
