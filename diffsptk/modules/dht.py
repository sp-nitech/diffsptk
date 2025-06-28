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
from ..utils.private import cas, check_size, filter_values, to
from .base import BaseFunctionalModule


class DiscreteHartleyTransform(BaseFunctionalModule):
    """Discrete Hartley transform module.

    Parameters
    ----------
    dht_length : int >= 1
        The DHT length, :math:`L`.

    dht_type : int in [1, 4]
        The DHT type.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        dht_length: int,
        dht_type: int = 2,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = dht_length

        _, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("W", tensors[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DHT to the input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            The input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            The DHT output.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> dht = diffsptk.DHT(4)
        >>> y = dht(x)
        >>> y
        tensor([ 3.0000, -1.4142, -1.0000, -1.4142])

        """
        check_size(x.size(-1), self.in_dim, "dimension of input")
        return self._forward(x, **self._buffers)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _, tensors = DiscreteHartleyTransform._precompute(
            x.size(-1), *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return DiscreteHartleyTransform._forward(x, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(dht_length: int, dht_type: int) -> None:
        if dht_length <= 0:
            raise ValueError("dht_length must be positive.")
        if not 1 <= dht_type <= 4:
            raise ValueError("dht_type must be in [1, 4].")

    @staticmethod
    def _precompute(
        dht_length: int,
        dht_type: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        DiscreteHartleyTransform._check(dht_length, dht_type)
        params = {"device": device, "dtype": torch.double}
        L = dht_length
        n = torch.arange(L, **params)
        k = torch.arange(L, **params)
        if dht_type in (2, 4):
            n += 0.5
        if dht_type in (3, 4):
            k += 0.5
        n *= 2 * torch.pi / L
        z = L**-0.5
        W = z * cas(k.unsqueeze(0) * n.unsqueeze(1))
        return None, None, (to(W, dtype=dtype),)

    @staticmethod
    def _forward(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, W)
