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

import torch.nn as nn

from .acorr import AutocorrelationAnalysis
from .levdur import PseudoLevinsonDurbinRecursion


class LinearPredictiveCodingAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lpc.html>`_
    for details. This module is a simple cascade of acorr and levdur.

    Parameters
    ----------
    acr_order : int >= 0 [scalar]
        Order of autocorrelation, :math:`M`.

    frame_length : int > M [scalar]
        Frame length, :math:`L`.

    out_format : ['K', 'a', 'Ka', 'K,a']
        `K` is gain and `a` is LPC coefficients. If this is `Ka`, the two output
        tensors are concatenated and return the tensor instead of the tuple.

    """

    def __init__(self, acr_order, frame_length, out_format="K,a"):
        super(LinearPredictiveCodingAnalysis, self).__init__()

        self.lpc = nn.Sequential(
            AutocorrelationAnalysis(acr_order, frame_length),
            PseudoLevinsonDurbinRecursion(out_format),
        )

    def forward(self, x):
        """Perform LPC analysis.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Framed waveform

        Returns
        -------
        a : Tensor or tuple[Tensor]
            Gain and/or LPC coefficients.

        Examples
        --------
        >>> x = torch.nrand(5)
        tensor([ 0.8226, -0.0284, -0.5715,  0.2127,  0.1217])
        >>> lpc = diffsptk.LPC(2, 5)
        >>> a = lpc(x)
        >>> a
        (tensor([0.8726]), tensor([0.1475, 0.5270]))

        """
        a = self.lpc(x)
        return a
