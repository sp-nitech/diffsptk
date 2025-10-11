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

from .base import BaseNonFunctionalModule
from .mglsadf import PseudoMGLSADigitalFilter


class PseudoInverseMGLSADigitalFilter(BaseNonFunctionalModule):
    """See :func:`~diffsptk.PseudoMGLSADigitalFilter` for details."""

    def __init__(self, filter_order: int, frame_period: int, **kwargs) -> None:
        super().__init__()

        self.mglsadf = PseudoMGLSADigitalFilter(filter_order, frame_period, **kwargs)

    def forward(self, y: torch.Tensor, mc: torch.Tensor) -> torch.Tensor:
        """Apply an inverse MGLSA digital filter.

        Parameters
        ----------
        y : Tensor [shape=(..., T)]
            The input signal.

        mc : Tensor [shape=(..., T/P, M+1)] or [shape=(..., T/P, N+M+1)]
            The mel-generalized cepstrum, not MLSA digital filter coefficients. Note
            that the mixed-phase case assumes that the coefficients are of the form
            c_{-N}, ..., c_{0}, ..., c_{M}, where M is the order of the minimum-phase
            part and N is the order of the maximum-phase part.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The residual signal.

        Examples
        --------
        >>> import diffsptk
        >>> import torch
        >>> imglsadf = diffsptk.IMLSA(1, frame_period=2)
        >>> y = diffsptk.step(3)
        >>> mc = torch.tensor([[0.3, 0.5], [-0.2, 0.1]])
        >>> x = imglsadf(y, mc)
        >>> x
        tensor([0.7408, 0.6659, 1.1176, 1.1048])

        """
        x = self.mglsadf(y, -mc)
        return x
