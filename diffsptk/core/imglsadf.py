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

from .mglsadf import PseudoMGLSADigitalFilter


class PseudoInverseMGLSADigitalFilter(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/imglsadf.html>`_
    for details. The exponential filter is approximated by the Taylor expansion.

    Parameters
    ----------
    filter_order : int >= 0 [scalar]
        Order of filter coefficients, :math:`M`.

    cep_order : int >= filter_order [scalar]
        Order of linear cepstrum.

    alpha : float [-1 < alpha < 1]
        Frequency warping factor, :math:`\\alpha`.

    gamma : float [-1 <= gamma <= 1]
        Gamma, :math:`\\gamma`.

    c : int >= 1 [scalar]
        Number of stages.

    taylor_order : int >= 0 [scalar]
        Order of Taylor series expansion, :math:`L`.

    frame_period : int >= 1 [scalar]
        Frame period, :math:`P`.

    ignore_gain : bool [scalar]
        If True, perform filtering without gain.

    """

    def __init__(
        self,
        filter_order,
        cep_order=200,
        alpha=0,
        gamma=0,
        c=None,
        taylor_order=50,
        frame_period=1,
        ignore_gain=False,
    ):
        super(PseudoInverseMGLSADigitalFilter, self).__init__()

        self.mglsadf = PseudoMGLSADigitalFilter(
            filter_order,
            cep_order=cep_order,
            alpha=alpha,
            gamma=gamma,
            c=c,
            taylor_order=taylor_order,
            frame_period=frame_period,
            ignore_gain=ignore_gain,
        )

    def forward(self, y, mc):
        """Apply an inverse MGLSA digital filter.

        Parameters
        ----------
        y : Tensor [shape=(..., T)]
            Audio signal.

        mc : Tensor [shape=(..., T/P, M+1)]
            Mel-generalized cepstrum, not MLSA digital filter coefficients.

        Returns
        -------
        x : Tensor [shape=(..., T)]
            Residual signal.

        Examples
        --------
        >>> M = 4
        >>> y = diffsptk.step(3)
        >>> mc = torch.randn(2, M + 1)
        >>> mc
        tensor([[ 0.8457,  1.5812,  0.1379,  1.6558,  1.4591],
                [-1.3714, -0.9669, -1.2025, -1.3683, -0.2352]])
        >>> imglsadf = diffsptk.PseudoInverseMGLSADigitalFilter(M, frame_period=2)
        >>> x = imglsadf(y.view(1, -1), mc.view(1, 2, M + 1))
        >>> x
        tensor([[ 0.4293,  1.0592,  7.9349, 14.9794]])

        """
        x = self.mglsadf(y, -mc)
        return x
