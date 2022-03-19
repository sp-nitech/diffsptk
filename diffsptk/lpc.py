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

from diffsptk.acorr import AutocorrelationAnalysis
from diffsptk.levdur import PseudoLevinsonDurbinRecursion


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

    See also
    --------
    diffsptk.acorr, diffsptk.levdur

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

        """
        a = self.lpc(x)
        return a
