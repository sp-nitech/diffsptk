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

import numpy as np
import torch
import torch.nn as nn

from ..misc.utils import numpy_to_torch


class MLSADigitalFilterCoefficientsToMelCepstrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/b2mc.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0 [scalar]
        Order of cepstrum, :math:`M`.

    alpha : float [-1 < alpha < 1]
        Frequency warping factor, :math:`\\alpha`.

    """

    def __init__(self, cep_order, alpha=0):
        super(MLSADigitalFilterCoefficientsToMelCepstrum, self).__init__()

        assert 0 <= cep_order
        assert abs(alpha) < 1

        # Make transform matrix.
        A = np.eye(cep_order + 1)
        np.fill_diagonal(A[:, 1:], alpha)

        self.register_buffer("A", numpy_to_torch(A.T))

    def forward(self, b):
        """Convert MLSA filter coefficients to mel-cepstrum.

        Parameters
        ----------
        b : Tensor [shape=(..., M+1)]
            MLSA filter coefficients.

        Returns
        -------
        mc : Tensor [shape=(..., M+1)]
            Mel-cepstral coefficients.

        Examples
        --------
        >>> b = diffsptk.ramp(4)
        >>> mc2b = diffsptk.MelCepstrumToMLSADigitalFilterCoefficients(4, 0.3)
        >>> b2mc = diffsptk.MLSADigitalFilterCoefficientsToMelCepstrum(4, 0.3)
        >>> b2 = mc2b(b2mc(b))
        >>> b2
        tensor([0.0000, 1.0000, 2.0000, 3.0000, 4.0000])

        """
        mc = torch.matmul(b, self.A)
        return mc
