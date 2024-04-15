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


class PolynomialToRoots(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/root_pol.html>`_
    for details.

    Parameters
    ----------
    order : int >= 1
        Order of polynomial.

    out_format : ['rectangular', 'polar']
        Output format.

    """

    def __init__(self, order, out_format="rectangular"):
        super().__init__()

        assert 1 <= order

        self.order = order
        self.formatter = self._formatter(out_format)
        self.register_buffer("eye", self._precompute(self.order))

    def forward(self, a):
        """Find roots of polynomial.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            Polynomial coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M)]
            Complex roots.

        Examples
        --------
        >>> a = torch.tensor([3, 4, 5])
        >>> root_pol = diffsptk.PolynomialToRoots(a.size(-1) - 1)
        >>> x = root_pol(a)
        >>> x
        tensor([[-0.6667+1.1055j, -0.6667-1.1055j]])

        """
        check_size(a.size(-1), self.order + 1, "order of polynomial")
        return self._forward(a, self.formatter, self.eye)

    @staticmethod
    def _forward(a, formatter, eye):
        if torch.any(a[..., 0] == 0):
            raise RuntimeError("Leading coefficient must be non-zero.")

        # Make companion matrix.
        a = -a[..., 1:] / a[..., :1]  # (..., M)
        E = eye.expand(a.size()[:-1] + eye.size())
        A = torch.cat((a.unsqueeze(-2), E), dim=-2)  # (..., M, M)

        # Find roots as eigenvalues.
        x, _ = torch.linalg.eig(A)
        x = formatter(x)
        return x

    @staticmethod
    def _func(a, out_format="rectangular"):
        formatter = PolynomialToRoots._formatter(out_format)
        eye = PolynomialToRoots._precompute(
            a.size(-1) - 1, dtype=a.dtype, device=a.device
        )
        return PolynomialToRoots._forward(a, formatter, eye)

    @staticmethod
    def _precompute(order, dtype=None, device=None):
        return torch.eye(order - 1, order, dtype=dtype, device=device)

    @staticmethod
    def _formatter(out_format):
        if out_format in (0, "rectangular"):
            return lambda x: x
        elif out_format in (1, "polar"):
            return lambda x: torch.complex(x.abs(), x.angle())
        raise ValueError(f"out_format {out_format} is not supported.")
