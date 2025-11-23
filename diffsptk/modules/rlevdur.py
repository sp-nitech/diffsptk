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

from ..typing import Precomputed
from ..utils.private import check_size, filter_values, remove_gain, to
from .base import BaseFunctionalModule


class ReverseLevinsonDurbin(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/rlevdur.html>`_
    for details.

    Parameters
    ----------
    lpc_order : int >= 0
        The order of the LPC coefficients, :math:`M`.

    n_fft : int >> M
        The number of FFT bins. Accurate conversion requires a large value.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        lpc_order: int,
        n_fft: int = 1024,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = lpc_order + 1

        _, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("phase_factors", tensors[0])

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """Solve a Yule-Walker linear system given the LPC coefficients.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            The gain and the LPC coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The autocorrelation.

        Examples
        --------
        >>> import diffsptk
        >>> acorr = diffsptk.Autocorrelation(5, 2)
        >>> levdur = diffsptk.LevinsonDurbin(2)
        >>> rlevdur = diffsptk.ReverseLevinsonDurbin(2)
        >>> x = diffsptk.ramp(1, 5) * 0.1
        >>> r = acorr(x)
        >>> r
        tensor([0.5500, 0.4000, 0.2600])
        >>> r2 = rlevdur(levdur(r))
        >>> r2
        tensor([0.5500, 0.4000, 0.2600])

        """
        check_size(a.size(-1), self.in_dim, "dimension of LPC coefficients")
        return self._forward(a, **self._buffers)

    @staticmethod
    def _func(a: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _, tensors = ReverseLevinsonDurbin._precompute(
            a.size(-1) - 1, *args, **kwargs, device=a.device, dtype=a.dtype
        )
        return ReverseLevinsonDurbin._forward(a, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(lpc_order: int, n_fft: int) -> None:
        if lpc_order < 0:
            raise ValueError("lpc_order must be non-negative.")
        if n_fft <= lpc_order + 1:
            raise ValueError("n_fft must be much larger than lpc_order.")

    @staticmethod
    def _precompute(
        lpc_order: int,
        n_fft: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        ReverseLevinsonDurbin._check(lpc_order, n_fft)
        n_freq = n_fft // 2 + 1
        omega = torch.linspace(0, np.pi, n_freq, device=device, dtype=torch.double)
        m = torch.arange(lpc_order + 1, device=device, dtype=torch.double)
        phase_factors = torch.exp(-1j * omega * m.unsqueeze(-1))
        return None, None, (to(phase_factors, dtype=dtype),)

    @staticmethod
    def _forward(a: torch.Tensor, phase_factors: torch.Tensor) -> torch.Tensor:
        M = a.size(-1) - 1
        K, a = remove_gain(a, return_gain=True)
        A = torch.sum(a.unsqueeze(-1) * phase_factors, dim=-2)
        r = torch.fft.irfft((K / A.abs()) ** 2)[..., : M + 1]
        return r
