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

from ..misc.utils import check_size
from ..misc.utils import get_values
from ..misc.utils import to
from .base import BaseFunctionalModule


class CompositeSinusoidalModelCoefficientsToAutocorrelation(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/csm2acr.html>`_
    for details.

    Parameters
    ----------
    acr_order : int >= 0
        The order of the autocorrelation, :math:`M`.

    References
    ----------
    .. [1] S. Sagayama et al., "Duality theory of composite sinusoidal modeling and
           linear prediction," *Proceedings of ICASSP*, pp. 1261-1264, 1986.

    """

    def __init__(self, acr_order):
        super().__init__()

        self.in_dim = acr_order + 1

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("ramp", tensors[0])

    def forward(self, c):
        """Convert CSM coefficients to autocorrelation.

        Parameters
        ----------
        c : Tensor [shape=(..., M+1)]
            The CSM coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The autocorrelation.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> acorr = diffsptk.Autocorrelation(5, 3)
        >>> acr2csm = diffsptk.AutocorrelationToCompositeSinusoidalModelCoefficients(3)
        >>> csm2acr = diffsptk.CompositeSinusoidalModelCoefficientsToAutocorrelation(3)
        >>> r = acorr(x)
        >>> r
        tensor([ 8.8894, -0.1102, -4.1748,  0.7501])
        >>> r2 = csm2acr(acr2csm(r))
        >>> r2
        tensor([ 8.8894, -0.1102, -4.1748,  0.7501])

        """
        check_size(c.size(-1), self.in_dim, "dimension of autocorrelation")
        return self._forward(c, **self._buffers)

    @staticmethod
    def _func(c, *args, **kwargs):
        _, _, tensors = (
            CompositeSinusoidalModelCoefficientsToAutocorrelation._precompute(
                c.size(-1) - 1, *args, **kwargs, device=c.device, dtype=c.dtype
            )
        )
        return CompositeSinusoidalModelCoefficientsToAutocorrelation._forward(
            c, *tensors
        )

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(acr_order):
        if acr_order <= 0 or acr_order % 2 == 0:
            raise ValueError("acr_order must be a positive odd number.")

    @staticmethod
    def _precompute(acr_order, device=None, dtype=None):
        CompositeSinusoidalModelCoefficientsToAutocorrelation._check(acr_order)
        ramp = torch.arange(acr_order + 1, device=device)
        return None, None, (to(ramp, dtype=dtype),)

    @staticmethod
    def _forward(c, ramp):
        w, m = torch.tensor_split(c, 2, dim=-1)
        a = m.unsqueeze(-2)  # (..., 1, (M+1)/2)
        b = torch.cos(w.unsqueeze(-1) * ramp)  # (..., (M+1)/2, M+1)
        r = torch.matmul(a, b).squeeze(-2)  # (..., M+1)
        return r
