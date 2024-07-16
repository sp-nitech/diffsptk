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

from .acorr import Autocorrelation
from .levdur import LevinsonDurbin


class LinearPredictiveCodingAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lpc.html>`_
    for details. Double precision is recommended.

    Parameters
    ----------
    frame_length : int > M
        Frame length, :math:`L`.

    lpc_order : int >= 0
        Order of LPC, :math:`M`.

    eps : float >= 0
        A small value to improve numerical stability.

    """

    def __init__(self, frame_length, lpc_order, eps=1e-6):
        super().__init__()

        self.lpc = nn.Sequential(
            Autocorrelation(frame_length, lpc_order),
            LevinsonDurbin(lpc_order, eps=eps),
        )

    def forward(self, x):
        """Compute LPC coefficients.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Framed waveform.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            Gain and LPC coefficients.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        tensor([ 0.8226, -0.0284, -0.5715,  0.2127,  0.1217])
        >>> lpc = diffsptk.LPC(5, 2)
        >>> a = lpc(x)
        >>> a
        tensor([0.8726, 0.1475, 0.5270])

        """
        return self.lpc(x)

    @staticmethod
    def _func(x, lpc_order, eps):
        r = Autocorrelation._func(x, lpc_order)
        a = LevinsonDurbin._func(r, eps)
        return a
