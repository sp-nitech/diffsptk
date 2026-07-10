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

from ..utils.private import check_size, filter_values, is_power_of_two, to
from .base import BaseFunctionalModule, Precomputed


class WalshHadamardTransform(BaseFunctionalModule):
    """Walsh-Hadamard Transform module.

    Parameters
    ----------
    wht_length : int >= 1
        The WHT length, :math:`L`, must be a power of 2.

    wht_type : ['sequency', 'natural', 'dyadic']
        The order of the coefficients in the Walsh matrix.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] K. Usha et al., "Generation of Walsh codes in two different orderings using
           4-bit Gray and Inverse Gray codes," *Indian Journal of Science and
           Technology*, vol. 5, no. 3, pp. 2341-2345, 2012.

    """

    _takes_input_size = True

    def __init__(
        self,
        wht_length: int,
        wht_type: str | int = "natural",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = wht_length

        self._register_precomputed(self._precompute(**filter_values(locals())))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply WHT to the input.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            The input.

        Returns
        -------
        out : Tensor [shape=(..., L)]
            The WHT output.

        Examples
        --------
        >>> import diffsptk
        >>> wht = diffsptk.WHT(4)
        >>> x = diffsptk.ramp(3)
        >>> y = wht(x)
        >>> y
        tensor([ 3., -1., -2.,  0.])
        >>> z = wht(y)
        >>> z
        tensor([0., 1., 2., 3.])

        """
        check_size(x.size(-1), self.in_dim, "dimension of input")
        return self._call_forward(x)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _p = WalshHadamardTransform._precompute(
            x.size(-1), *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return WalshHadamardTransform._apply_precomputed(_p, x=x)

    @staticmethod
    def _check(wht_length: int) -> None:
        if wht_length <= 0 or not is_power_of_two(wht_length):
            raise ValueError("wht_length must be a power of 2.")

    @staticmethod
    def _precompute(
        wht_length: int,
        wht_type: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        from scipy.linalg import hadamard

        WalshHadamardTransform._check(wht_length)
        L = wht_length
        z = 2 ** -(np.log2(L) / 2)
        W = hadamard(L)
        if wht_type in (1, "sequency"):
            sign_changes = np.sum(np.abs(np.diff(W, axis=1)), axis=1)
            W = W[np.argsort(sign_changes)]
        elif wht_type in (2, "natural"):
            pass
        elif wht_type in (3, "dyadic"):
            gray_bits = [
                [int(x) for x in np.binary_repr(i, width=int(np.log2(L)))]
                for i in range(L)
            ]
            binary_bits = np.bitwise_xor.accumulate(gray_bits, axis=1)
            permutation = [int("".join(row), 2) for row in binary_bits.astype(str)]
            sign_changes = np.sum(np.abs(np.diff(W, axis=1)), axis=1)
            W = W[np.argsort(sign_changes)][permutation]
        else:
            raise ValueError(f"wht_type {wht_type} is not supported.")
        return Precomputed(tensors={"W": to(W * z, device=device, dtype=dtype)})

    @staticmethod
    def _forward(x: torch.Tensor, *, W: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, W)
