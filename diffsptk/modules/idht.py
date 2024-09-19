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
from .dht import DiscreteHartleyTransform as DHT


class InverseDiscreteHartleyTransform(nn.Module):
    """This is the opposite module to :func:`~diffsptk.DiscreteHartleyTransform`.

    Parameters
    ----------
    dht_length : int >= 1
        DHT length, :math:`L`.

    dht_type : int in [1, 4]
        DHT type.

    """

    def __init__(self, dht_length, dht_type=2):
        super().__init__()

        assert 1 <= dht_length
        assert 1 <= dht_type <= 4

        self.dht_length = dht_length
        self.register_buffer("W", self._precompute(dht_length, dht_type))

    def forward(self, y):
        """Apply inverse DHT to input.

        Parameters
        ----------
        y : Tensor [shape=(..., L)]
            Input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            Inverse DHT output.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> dht = diffsptk.DHT(4)
        >>> idht = diffsptk.IDHT(4)
        >>> x2 = idht(dht(x))
        >>> x2
        tensor([5.9605e-08, 1.0000e+00, 2.0000e+00, 3.0000e+00])

        """
        check_size(y.size(-1), self.dht_length, "dimension of input")
        return self._forward(y, self.W)

    @staticmethod
    def _forward(y, W):
        return torch.matmul(y, W)

    @staticmethod
    def _func(y, dht_type):
        W = InverseDiscreteHartleyTransform._precompute(
            y.size(-1), dht_type, dtype=y.dtype, device=y.device
        )
        return InverseDiscreteHartleyTransform._forward(y, W)

    @staticmethod
    def _precompute(dht_length, dht_type, dtype=None, device=None):
        type2type = {1: 1, 2: 3, 3: 2, 4: 4}
        return DHT._precompute(
            dht_length, type2type[dht_type], dtype=dtype, device=device
        )
