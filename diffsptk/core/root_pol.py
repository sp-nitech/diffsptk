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
import torch.nn.functional as F

from ..misc.utils import check_size
from ..misc.utils import numpy_to_torch
from ..misc.utils import vander


class DurandKernerMethod(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/root_pol.html>`_
    for details.

    order : int >= 1 [scalar]
        Order of coefficients.

    n_iter : int >= 1 [scalar]
        Number of iterations.

    eps : float >= 0 [scalar]
        Convergence threshold.

    out_format : ['rectangular', 'polar']
        Output format.

    """

    def __init__(self, order, n_iter=100, eps=1e-14, out_format="rectangular"):
        super(DurandKernerMethod, self).__init__()

        self.order = order
        self.n_iter = n_iter
        self.eps = eps

        assert 1 <= self.order
        assert 1 <= self.n_iter
        assert 0 <= self.eps

        if out_format == 0 or out_format == "rectangular":
            self.convert = lambda x: x
        elif out_format == 1 or out_format == "polar":
            self.convert = lambda x: torch.complex(x.abs(), x.angle())
        else:
            raise ValueError(f"out_format {out_format} is not supported")

        ramp = np.arange(order + 1)
        exponent = 1 / ramp[2:]
        self.register_buffer("exponent", numpy_to_torch(exponent))

        angle = ramp[:-1] * np.pi / (order / 2) + np.pi / (order * 2)
        self.register_buffer("sin", numpy_to_torch(np.sin(angle)))
        self.register_buffer("cos", numpy_to_torch(np.cos(angle)))
        self.register_buffer("eye", torch.eye(order).bool())

    def forward(self, a):
        """Find roots of equations.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            Polynomial coefficients.

        Returns
        -------
        x : Tensor [shape=(..., M)]
            Complex roots.

        is_converged : Tensor [shape=(...,)]
            True if convergence is reached.

        Examples
        --------
        >>> a = torch.tensor([3, 4, 5])
        >>> root_pol = diffsptk.DurandKernerMethod(a.size(-1) - 1)
        >>> x, is_converged = root_pol(a)
        >>> x
        tensor([[-0.6667+1.1055j, -0.6667-1.1055j]])
        >>> is_converged
        tensor([True])

        """
        check_size(a.size(-1), self.order + 1, "dimension of coefficients")

        a = a[..., 1:] / a[..., :1]  # (..., M)
        radius, _ = torch.max(
            2 * torch.pow(a[..., 1:].abs(), self.exponent),
            dim=-1,
            keepdim=True,
        )
        center = -a[..., :1] / self.order
        x = torch.complex(
            center + radius * self.cos,
            center + radius * self.sin,
        )
        a = F.pad(a, (1, 0), value=1)
        a = a.unsqueeze(-1).to(x.dtype)

        for _ in range(self.n_iter):
            y = x
            for m in range(self.order):
                xm = x[..., m : m + 1]
                v = vander(xm, N=self.order + 1)
                numer = torch.matmul(v, a).squeeze(-1)

                w = (xm - x) + self.eye[m : m + 1]
                denom = w.prod(dim=-1, keepdim=True)

                delta = numer / denom
                x = x - delta * self.eye[m : m + 1]

            if (y - x).abs().max() <= self.eps:
                break

        is_converged = torch.max((y - x).abs(), dim=-1)[0] <= self.eps
        x = self.convert(x)

        return x, is_converged
