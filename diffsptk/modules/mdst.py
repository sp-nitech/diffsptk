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

from torch import nn

from ..utils.private import get_values
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

    """

    def __init__(self, frame_length, window="sine"):
        super().__init__()

        self.values, layers, _ = self._precompute(*get_values(locals()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
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
    def _func(*args, **kwargs):
        return MDCT._func(*args, **kwargs, transform="sine")

    @staticmethod
    def _takes_input_size():
        return False

    @staticmethod
    def _check(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _precompute(frame_length, window):
        return MDCT._precompute(frame_length, window, transform="sine")

    @staticmethod
    def _forward(*args, **kwargs):
        return MDCT._forward(*args, **kwargs)
