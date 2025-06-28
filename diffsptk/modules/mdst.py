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

from ..typing import Precomputed
from ..utils.private import filter_values
from .base import BaseFunctionalModule
from .mdct import ModifiedDiscreteCosineTransform as MDCT


class ModifiedDiscreteSineTransform(BaseFunctionalModule):
    """This module is a simple cascade of framing, windowing, and modified DST.

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

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        frame_length: int,
        window: str = "sine",
        learnable: bool | list[str] = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.values, layers, _ = self._precompute(**filter_values(locals()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute modified discrete sine transform.

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
        >>> mdst = diffsptk.MDST(frame_length=4)
        >>> y = mdst(x)
        >>> y
        tensor([[-0.2071, -0.5000],
                [ 1.5858,  0.4142],
                [ 4.6213, -1.9142]])

        """
        return self._forward(x, *self.values, *self.layers)

    @staticmethod
    def _func(*args, **kwargs) -> torch.Tensor:
        return MDCT._func(*args, **kwargs, transform="sine")

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(*args, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def _precompute(*args, **kwargs) -> Precomputed:
        return MDCT._precompute(*args, **kwargs, transform="sine")

    @staticmethod
    def _forward(*args, **kwargs) -> torch.Tensor:
        return MDCT._forward(*args, **kwargs)
