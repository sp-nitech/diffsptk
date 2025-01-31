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

from .mglsadf import PseudoMGLSADigitalFilter


class PseudoInverseMGLSADigitalFilter(nn.Module):
    """See :func:`~diffsptk.PseudoMGLSADigitalFilter` for details."""

    def __init__(self, filter_order, frame_period, **kwargs):
        super().__init__()

        self.mglsadf = PseudoMGLSADigitalFilter(filter_order, frame_period, **kwargs)

    def forward(self, y, mc):
        """Apply an inverse MGLSA digital filter.

        Parameters
        ----------
        y : Tensor [shape=(..., T)]
            Audio signal.

        mc : Tensor [shape=(..., T/P, M+1)] or [shape=(..., T/P, N+M+1)]
            Mel-generalized cepstrum, not MLSA digital filter coefficients. Note that
            the mixed-phase case assumes that the coefficients are of the form
            c_{-N}, ..., c_{0}, ..., c_{M}, where M is the order of the minimum-phase
            part and N is the order of the maximum-phase part.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            Residual signal.

        Examples
        --------
        >>> M = 4
        >>> y = diffsptk.step(3)
        >>> mc = diffsptk.nrand(2, M)
        >>> mc
        tensor([[ 0.8457,  1.5812,  0.1379,  1.6558,  1.4591],
                [-1.3714, -0.9669, -1.2025, -1.3683, -0.2352]])
        >>> imglsadf = diffsptk.IMLSA(M, frame_period=2)
        >>> x = imglsadf(y.view(1, -1), mc.view(1, 2, M + 1))
        >>> x
        tensor([[ 0.4293,  1.0592,  7.9349, 14.9794]])

        """
        x = self.mglsadf(y, -mc)
        return x
