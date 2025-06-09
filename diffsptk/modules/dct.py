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

from ..typing import Precomputed
from ..utils.private import check_size, get_values, plateau, to
from .base import BaseFunctionalModule


class DiscreteCosineTransform(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/dct.html>`_
    for details.

    Parameters
    ----------
    dct_length : int >= 1
        The DCT length, :math:`L`.

    dct_type : int in [1, 4]
        The DCT type.

    """

    def __init__(self, dct_length: int, dct_type: int = 2) -> None:
        super().__init__()

        self.in_dim = dct_length

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("W", tensors[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DCT to the input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            The input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            The DCT output.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> dct = diffsptk.DCT(4)
        >>> y = dct(x)
        >>> y
        tensor([ 3.0000, -2.2304,  0.0000, -0.1585])

        """
        check_size(x.size(-1), self.in_dim, "dimension of input")
        return self._forward(x, **self._buffers)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _, tensors = DiscreteCosineTransform._precompute(
            x.size(-1), *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return DiscreteCosineTransform._forward(x, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(dct_length: int, dct_type: int) -> None:
        if dct_length <= 0:
            raise ValueError("dct_length must be positive.")
        if not 1 <= dct_type <= 4:
            raise ValueError("dct_type must be in [1, 4].")

    @staticmethod
    def _precompute(
        dct_length: int,
        dct_type: int = 2,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        DiscreteCosineTransform._check(dct_length, dct_type)
        params = {"device": device, "dtype": torch.double}
        L = dct_length
        n = torch.arange(L, **params)
        k = torch.arange(L, **params)
        if dct_type in (2, 4):
            n += 0.5
        if dct_type in (3, 4):
            k += 0.5
        n *= torch.pi / ((L - 1) if dct_type == 1 else L)

        if dct_type == 1:
            c = (1 / 2) ** 0.5
            z0 = plateau(L, c, 1, c, **params)
            z1 = plateau(L, 1, 2, 1, **params)
            z = z0.unsqueeze(0) * torch.sqrt(z1 / (L - 1)).unsqueeze(1)
        elif dct_type == 2:
            z = plateau(L, 1, 2, **params)
            z = torch.sqrt(z / L).unsqueeze(0)
        elif dct_type == 3:
            z = plateau(L, 1, 2, **params)
            z = torch.sqrt(z / L).unsqueeze(1)
        elif dct_type == 4:
            z = (2 / L) ** 0.5
        else:
            raise ValueError(f"dct_type {dct_type} is not supported.")

        W = z * torch.cos(k.unsqueeze(0) * n.unsqueeze(1))
        return None, None, (to(W, dtype=dtype),)

    @staticmethod
    def _forward(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, W)
