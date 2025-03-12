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
from .imdct import InverseModifiedDiscreteCosineTransform as IMDCT


class InverseModifiedDiscreteSineTransform(BaseFunctionalModule):
    """This is the opposite module to :func:`~diffsptk.ModifiedDiscreteSineTransform`.

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

    def forward(self, y, out_length=None):
        """Compute inverse modified discrete sine transform.

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
        >>> mdst_params = {"frame_length": 4}
        >>> mdst = diffsptk.MDST(**mdst_params)
        >>> imdst = diffsptk.IMDST(**mdst_params)
        >>> y = imdst(mdst(x))
        >>> y
        tensor([-8.9407e-08, 1.0000e+00, 2.0000e+00, 3.0000e+00])

        """
        return self._forward(y, out_length, *self.values, *self.layers)

    @staticmethod
    def _func(*args, **kwargs):
        return IMDCT._func(*args, **kwargs, transform="sine")

    @staticmethod
    def _takes_input_size():
        return False

    @staticmethod
    def _check(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _precompute(frame_length, window):
        return IMDCT._precompute(frame_length, window, transform="sine")

    @staticmethod
    def _forward(*args, **kwargs):
        return IMDCT._forward(*args, **kwargs)
