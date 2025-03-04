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

from ..utils.private import check_size
from ..utils.private import get_values
from .base import BaseFunctionalModule
from .dst import DiscreteSineTransform as DST


class InverseDiscreteSineTransform(BaseFunctionalModule):
    """This is the opposite module to :func:`~diffsptk.DiscreteSineTransform`.

    Parameters
    ----------
    dst_length : int >= 1
        The DST length, :math:`L`.

    dst_type : int in [1, 4]
        The DST type.

    """

    def __init__(self, dst_length, dst_type=2):
        super().__init__()

        self.in_dim = dst_length

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("W", tensors[0])

    def forward(self, y):
        """Apply inverse DST to the input.

        Parameters
        ----------
        y : Tensor [shape=(..., L)]
            The input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            The inverse DST output.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> dst = diffsptk.DST(4)
        >>> idst = diffsptk.IDST(4)
        >>> x2 = idst(dst(x))
        >>> x2
        tensor([1.1921e-07, 1.0000e+00, 2.0000e+00, 3.0000e+00])

        """
        check_size(y.size(-1), self.in_dim, "dimension of input")
        return self._forward(y, **self._buffers)

    @staticmethod
    def _func(y, *args, **kwargs):
        _, _, tensors = InverseDiscreteSineTransform._precompute(
            y.size(-1), *args, **kwargs, device=y.device, dtype=y.dtype
        )
        return InverseDiscreteSineTransform._forward(y, *tensors)

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(*args, **kwargs):
        DST._check(*args, **kwargs)

    @staticmethod
    def _precompute(dst_length, dst_type, device=None, dtype=None):
        type2type = {1: 1, 2: 3, 3: 2, 4: 4}
        return DST._precompute(
            dst_length, type2type[dst_type], device=device, dtype=dtype
        )

    @staticmethod
    def _forward(y, W):
        return torch.matmul(y, W)
