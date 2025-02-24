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
        The order of the polynomial.

    out_format : ['rectangular', 'polar']
        The output format.

    """

    def __init__(self, order, out_format="rectangular"):
        super().__init__()

        self.in_dim = order + 1

        self.values, tensors = self._precompute(order, out_format)
        self.register_buffer("eye", tensors[0])

    def forward(self, a):
        """Find the roots of the input polynomial.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            The polynomial coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M)]
            The complex roots.

        Examples
        --------
        >>> a = torch.tensor([3, 4, 5])
        >>> root_pol = diffsptk.PolynomialToRoots(a.size(-1) - 1)
        >>> x = root_pol(a)
        >>> x
        tensor([[-0.6667+1.1055j, -0.6667-1.1055j]])

        """
        check_size(a.size(-1), self.in_dim, "order of polynomial")
        return self._forward(a, *self.values, **self._buffers)

    @staticmethod
    def _func(a, *args, **kwargs):
        values, tensors = PolynomialToRoots._precompute(
            a.size(-1) - 1, *args, **kwargs, dtype=a.dtype, device=a.device
        )
        return PolynomialToRoots._forward(a, *values, *tensors)

    @staticmethod
    def _check(order):
        if order <= 0:
            raise ValueError("order must be positive.")

    @staticmethod
    def _precompute(order, out_format="rectangular", dtype=None, device=None):
        PolynomialToRoots._check(order)

        if out_format in (0, "rectangular"):
            formatter = lambda x: x
        elif out_format in (1, "polar"):
            formatter = lambda x: torch.complex(x.abs(), x.angle())
        else:
            raise ValueError(f"out_format {out_format} is not supported.")

        eye = torch.eye(order - 1, order, dtype=dtype, device=device)
        return (formatter,), (eye,)

    @staticmethod
    def _forward(a, formatter, eye):
        if torch.any(a[..., 0] == 0):
            raise ValueError("Leading coefficient must be non-zero.")

        # Make companion matrix.
        a = -a[..., 1:] / a[..., :1]  # (..., M)
        E = eye.expand(a.size()[:-1] + eye.size())
        A = torch.cat((a.unsqueeze(-2), E), dim=-2)  # (..., M, M)

        # Find roots as eigenvalues.
        x, _ = torch.linalg.eig(A)
        x = formatter(x)
        return x
