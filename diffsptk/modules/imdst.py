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

from .imdct import InverseModifiedDiscreteCosineTransform as IMDST


class InverseModifiedDiscreteSineTransform(nn.Module):
    """This is the opposite module to :func:`~diffsptk.ModifiedDiscreteSineTransform`.

    Parameters
    ----------
    frame_length : int >= 2
        Frame length, :math:`L`.

    window : ['sine', 'vorbis', 'kbd', 'rectangular']
        Window type.

    """

    def __init__(self, frame_length, window="sine"):
        super().__init__()

        self.imdst = IMDST(frame_length, window, transform="sine")

    def forward(self, y, out_length=None):
        """Compute inverse modified discrete sine transform.

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
        >>> mdst_params = {"frame_length": 4}
        >>> mdst = diffsptk.MDST(**mdst_params)
        >>> imdst = diffsptk.IMDST(**mdst_params)
        >>> y = imdst(mdst(x))
        >>> y
        tensor([-8.9407e-08, 1.0000e+00, 2.0000e+00, 3.0000e+00])

        """
        return self.imdst(y, out_length=out_length)

    @staticmethod
    def _func(y, out_length, frame_length, window):
        return IMDST._func(y, out_length, frame_length, window, transform="sine")
