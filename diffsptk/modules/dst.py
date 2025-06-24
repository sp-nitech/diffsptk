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


class DiscreteSineTransform(BaseFunctionalModule):
    """Discrete sine transform module.

    Parameters
    ----------
    dst_length : int >= 1
        The DST length, :math:`L`.

    dst_type : int in [1, 4]
        The DST type.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        dst_length: int,
        dst_type: int = 2,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = dst_length

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("W", tensors[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DST to the input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            The input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            The DST output.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> dst = diffsptk.DST(4)
        >>> y = dst(x)
        >>> y
        tensor([ 2.7716, -2.0000,  1.1481, -1.0000])

        """
        check_size(x.size(-1), self.in_dim, "dimension of input")
        return self._forward(x, **self._buffers)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _, tensors = DiscreteSineTransform._precompute(
            x.size(-1), *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return DiscreteSineTransform._forward(x, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(dst_length: int, dst_type: int) -> None:
        if dst_length <= 0:
            raise ValueError("dst_length must be positive.")
        if not 1 <= dst_type <= 4:
            raise ValueError("dst_type must be in [1, 4].")

    @staticmethod
    def _precompute(
        dst_length: int,
        dst_type: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        DiscreteSineTransform._check(dst_length, dst_type)
        params = {"device": device, "dtype": torch.double}
        L = dst_length
        n = torch.arange(1, L + 1, **params)
        k = torch.arange(1, L + 1, **params)
        if dst_type in (2, 4):
            n -= 0.5
        if dst_type in (3, 4):
            k -= 0.5
        n *= torch.pi / ((L + 1) if dst_type == 1 else L)

        if dst_type == 1:
            z = (2 / (L + 1)) ** 0.5
        elif dst_type == 2:
            z = plateau(L, 2, 2, 1, **params)
            z = torch.sqrt(z / L).unsqueeze(0)
        elif dst_type == 3:
            z = plateau(L, 2, 2, 1, **params)
            z = torch.sqrt(z / L).unsqueeze(1)
        elif dst_type == 4:
            z = (2 / L) ** 0.5
        else:
            raise ValueError(f"dst_type {dst_type} is not supported.")

        W = z * torch.sin(k.unsqueeze(0) * n.unsqueeze(1))
        return None, None, (to(W, dtype=dtype),)

    @staticmethod
    def _forward(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, W)
