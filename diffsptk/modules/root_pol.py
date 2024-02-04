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
import torch.nn as nn

from ..misc.utils import check_mode
from ..misc.utils import check_size

_convert = {
    "rectangular": lambda x: x,
    "polar": lambda x: torch.complex(x.abs(), x.angle()),
}


class PolynomialToRoots(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/root_pol.html>`_
    for details.

    order : int >= 1
        Order of polynomial.

    out_format : ['rectangular', 'polar']
        Output format.

    """

    def __init__(self, order, out_format="rectangular", stateful=True):
        super(PolynomialToRoots, self).__init__()

        self.order = order
        self.out_format = check_mode(out_format, _convert)

        assert 1 <= self.order

        if stateful:
            self.register_buffer("eye", self._make_eye(self.order))

    def forward(self, a):
        """Find roots of polynomial.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            Polynomial coefficients.

        Returns
        -------
        Tensor [shape=(..., M)]
            Complex roots.

        Examples
        --------
        >>> a = torch.tensor([3, 4, 5])
        >>> root_pol = diffsptk.PolynomialToRoots(a.size(-1) - 1)
        >>> x = root_pol(a)
        >>> x
        tensor([[-0.6667+1.1055j, -0.6667-1.1055j]])

        """
        check_size(a.size(-1), self.order + 1, "dimension of coefficients")
        return self._forward(a, self.out_format, eye=getattr(self, "eye", None))

    @staticmethod
    def _forward(a, out_format, eye=None):
        if torch.any(a[..., 0] == 0):
            raise RuntimeError("leading coefficient must be non-zero")

        # Make companion matrix.
        a = -a[..., 1:] / a[..., :1]  # (..., M)
        if eye is None:
            eye = PolynomialToRoots._make_eye(
                a.size(-1), dtype=a.dtype, device=a.device
            )
        E = eye.expand(a.size()[:-1] + eye.size())
        A = torch.cat((a.unsqueeze(-2), E), dim=-2)  # (..., M, M)

        # Find roots as eigenvalues.
        x, _ = torch.linalg.eig(A)
        try:
            x = _convert[out_format](x)
        except KeyError:
            raise ValueError(f"out_format: {out_format} is not supported.")
        return x

    @staticmethod
    def _make_eye(order, dtype=None, device=None):
        return torch.eye(order - 1, order, dtype=dtype, device=device)
