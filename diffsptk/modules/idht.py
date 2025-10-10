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
from ..utils.private import check_size, filter_values
from .base import BaseFunctionalModule
from .dht import DiscreteHartleyTransform as DHT


class InverseDiscreteHartleyTransform(BaseFunctionalModule):
    """This is the opposite module to :func:`~diffsptk.DiscreteHartleyTransform`.

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

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse DHT to the input.

        Parameters
        ----------
        y : Tensor [shape=(..., L)]
            The input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            The inverse DHT output.

        Examples
        --------
        >>> import diffsptk
        >>> dht = diffsptk.DHT(4)
        >>> idht = diffsptk.IDHT(4)
        >>> x = diffsptk.ramp(1, 4)
        >>> x2 = idht(dht(x))
        >>> x2
        tensor([1., 2., 3., 4.])

        """
        check_size(y.size(-1), self.in_dim, "dimension of input")
        return self._forward(y, **self._buffers)

    @staticmethod
    def _func(y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _, tensors = InverseDiscreteHartleyTransform._precompute(
            y.size(-1), *args, **kwargs, device=y.device, dtype=y.dtype
        )
        return InverseDiscreteHartleyTransform._forward(y, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(*args, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def _precompute(
        dht_length: int,
        dht_type: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        type2type = {1: 1, 2: 3, 3: 2, 4: 4}
        return DHT._precompute(
            dht_length, type2type[dht_type], device=device, dtype=dtype
        )

    @staticmethod
    def _forward(y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.matmul(y, W)
