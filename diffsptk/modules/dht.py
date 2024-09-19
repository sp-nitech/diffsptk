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

from ..misc.utils import cas
from ..misc.utils import check_size
from ..misc.utils import to


class DiscreteHartleyTransform(nn.Module):
    """Discrete Hartley transform module.

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

    def forward(self, x):
        """Apply DHT to input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            Input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            DHT output.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> dht = diffsptk.DHT(4)
        >>> y = dht(x)
        >>> y
        tensor([ 3.0000, -1.4142, -1.0000, -1.4142])

        """
        check_size(x.size(-1), self.dht_length, "dimension of input")
        return self._forward(x, self.W)

    @staticmethod
    def _forward(x, W):
        return torch.matmul(x, W)

    @staticmethod
    def _func(x, dht_type):
        W = DiscreteHartleyTransform._precompute(
            x.size(-1), dht_type, dtype=x.dtype, device=x.device
        )
        return DiscreteHartleyTransform._forward(x, W)

    @staticmethod
    def _precompute(length, dht_type, dtype=None, device=None):
        L = length
        n = torch.arange(L, dtype=torch.double, device=device)
        k = torch.arange(L, dtype=torch.double, device=device)
        if dht_type == 2 or dht_type == 4:
            n += 0.5
        if dht_type == 3 or dht_type == 4:
            k += 0.5
        n *= 2 * torch.pi / L
        z = L**-0.5
        W = z * cas(k.unsqueeze(0) * n.unsqueeze(1))
        return to(W, dtype=dtype)
