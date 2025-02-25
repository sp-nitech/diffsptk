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

    """

    def __init__(self, dht_length, dht_type=2, device=None, dtype=None):
        super().__init__()

        self.in_dim = dht_length

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("W", tensors[0])

    def forward(self, y):
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
        >>> x = diffsptk.ramp(3)
        >>> dht = diffsptk.DHT(4)
        >>> idht = diffsptk.IDHT(4)
        >>> x2 = idht(dht(x))
        >>> x2
        tensor([5.9605e-08, 1.0000e+00, 2.0000e+00, 3.0000e+00])

        """
        check_size(y.size(-1), self.in_dim, "dimension of input")
        return self._forward(y, **self._buffers)

    @staticmethod
    def _func(y, *args, **kwargs):
        _, _, tensors = InverseDiscreteHartleyTransform._precompute(
            y.size(-1), *args, **kwargs, device=y.device, dtype=y.dtype
        )
        return InverseDiscreteHartleyTransform._forward(y, *tensors)

    @staticmethod
    def _check(*args, **kwargs):
        DHT._check(*args, **kwargs)

    @staticmethod
    def _precompute(dht_length, dht_type, device=None, dtype=None):
        type2type = {1: 1, 2: 3, 3: 2, 4: 4}
        return DHT._precompute(
            dht_length, type2type[dht_type], device=device, dtype=dtype
        )

    @staticmethod
    def _forward(y, W):
        return torch.matmul(y, W)
