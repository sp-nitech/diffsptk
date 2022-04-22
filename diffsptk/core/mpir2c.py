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

from .mgc2mgc import MelGeneralizedCepstrumToMelGeneralizedCepstrum


class MinimumPhaseImpulseResponseToCepstrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mpir2c.html>`_
    for details. This module may be slow due to recursive computation.

    Parameters
    ----------
    cep_order : int >= 0 [scalar]
        Order of cepstrum, :math:`M`.

    impulse_response_length : int >= 1 [scalar]
        Length of impulse response, :math:`N`.

    """

    def __init__(self, cep_order, impulse_response_length):
        super(MinimumPhaseImpulseResponseToCepstrum, self).__init__()

        self.ir2c = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
            impulse_response_length - 1,
            cep_order,
            in_gamma=1,
            out_gamma=0,
            in_mul=True,
            out_mul=False,
        )

    def forward(self, h):
        """Convert impulse response to cepstrum.

        Parameters
        ----------
        h : Tensor [shape=(..., N)]
            Truncated impulse response.

        Returns
        -------
        c : Tensor [shape=(..., M+1)]
            Cepstral coefficients.

        Examples
        --------
        >>> h = diffsptk.ramp(4, 0, -1)
        >>> mpir2c = diffsptk.MinimumPhaseImpulseResponseToCepstrum(3, 5)
        >>> c = mpir2c(h)
        >>> c
        tensor([1.3863, 0.7500, 0.2188, 0.0156])

        """
        c = self.ir2c(h)
        return c
