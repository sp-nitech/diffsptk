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
from torch import nn

from ..misc.utils import check_size


class LogAreaRatioToParcorCoefficients(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lar2par.html>`_
    for details.

    Parameters
    ----------
    par_order : int >= 0
        Order of PARCOR, :math:`M`.

    """

    def __init__(self, par_order):
        super().__init__()

        assert 0 <= par_order

        self.par_order = par_order

    def forward(self, g):
        """Convert LAR to PARCOR.

        Parameters
        ----------
        g : Tensor [shape=(..., M+1)]
            Log area ratio.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            PARCOR coefficients.

        Examples
        --------
        >>> g = diffsptk.ramp(1, 4) * 0.1
        >>> lar2par = diffsptk.LogAreaRatioToParcorCoefficients(3)
        >>> k = lar2par(g)
        >>> k
        tensor([0.1000, 0.0997, 0.1489, 0.1974])

        """
        check_size(g.size(-1), self.par_order + 1, "dimension of parcor")
        return self._forward(g)

    @staticmethod
    def _forward(g):
        K, g = torch.split(g, [1, g.size(-1) - 1], dim=-1)
        k = torch.cat((K, torch.tanh(0.5 * g)), dim=-1)
        return k

    _func = _forward
