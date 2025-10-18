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
import torch.nn.functional as F

from ..typing import Callable, Precomputed
from ..utils.private import TAU, check_size, filter_values, to_3d
from .base import BaseFunctionalModule
from .pol_root import RootsToPolynomial


class LineSpectralPairsToLinearPredictiveCoefficients(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lsp2lpc.html>`_
    for details.

    Parameters
    ----------
    lpc_order : int >= 0
        The order of the LPC, :math:`M`.

    log_gain : bool
        If True, assume the input gain is in logarithmic scale.

    sample_rate : int >= 1 or None
        The sample rate in Hz.

    in_format : ['radian', 'cycle', 'khz', 'hz']
        The input format.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        lpc_order: int,
        log_gain: bool = False,
        sample_rate: int | None = None,
        in_format: str | int = "radian",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = lpc_order + 1

        self.values, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("kernel_p", tensors[0])
        self.register_buffer("kernel_q", tensors[1])

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Convert LSP to LPC.

        Parameters
        ----------
        w : Tensor [shape=(..., M+1)]
            The LSP frequencies.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The LPC coefficients.

        Examples
        --------
        >>> import diffsptk
        >>> lsp2lpc = diffsptk.LineSpectralPairsToLinearPredictiveCoefficients(3)
        >>> w = diffsptk.ramp(3)
        >>> a = lsp2lpc(w)
        >>> a
        tensor([ 0.0000,  0.8658, -0.0698,  0.0335])

        """
        check_size(w.size(-1), self.in_dim, "dimension of LSP")
        return self._forward(w, *self.values, **self._buffers)

    @staticmethod
    def _func(w: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, tensors = (
            LineSpectralPairsToLinearPredictiveCoefficients._precompute(
                w.size(-1) - 1, *args, **kwargs, device=w.device, dtype=w.dtype
            )
        )
        return LineSpectralPairsToLinearPredictiveCoefficients._forward(
            w, *values, *tensors
        )

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(
        lpc_order: int, log_gain: bool, sample_rate: int | None, in_format: str | int
    ) -> None:
        if lpc_order < 0:
            raise ValueError("lpc_order must be non-negative.")
        if in_format in (2, 3, "hz", "khz") and (
            sample_rate is None or sample_rate <= 0
        ):
            raise ValueError("sample_rate must be positive.")

    @staticmethod
    def _precompute(
        lpc_order: int,
        log_gain: bool,
        sample_rate: int | None,
        in_format: str | int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        LineSpectralPairsToLinearPredictiveCoefficients._check(
            lpc_order, log_gain, sample_rate, in_format
        )

        if in_format in (0, "radian"):
            formatter = lambda x: x
        elif in_format in (1, "cycle"):
            formatter = lambda x: x * TAU
        elif in_format in (2, "khz"):
            formatter = lambda x: x * (TAU / sample_rate * 1000)
        elif in_format in (3, "hz"):
            formatter = lambda x: x * (TAU / sample_rate)
        else:
            raise ValueError(f"in_format {in_format} is not supported.")

        params = {"device": device, "dtype": dtype}
        if lpc_order % 2 == 0:
            kernel_p = torch.tensor([-1.0, 1.0], **params)
            kernel_q = torch.tensor([1.0, 1.0], **params)
        else:
            kernel_p = torch.tensor([-1.0, 0.0, 1.0], **params)
            kernel_q = torch.tensor([0.0, 1.0, 0.0], **params)
        kernel_p = kernel_p.view(1, 1, -1)
        kernel_q = kernel_q.view(1, 1, -1)

        return (log_gain, formatter), None, (kernel_p, kernel_q)

    @staticmethod
    def _forward(
        w: torch.Tensor,
        log_gain: bool,
        formatter: Callable,
        kernel_p: torch.Tensor,
        kernel_q: torch.Tensor,
    ) -> torch.Tensor:
        M = w.size(-1) - 1
        K, w = torch.split(w, [1, M], dim=-1)
        if log_gain:
            K = torch.exp(K)

        if M == 0:
            return K

        w = formatter(w)
        z = torch.exp(1j * to_3d(w))
        p = z[..., 1::2]
        q = z[..., 0::2]
        if M == 1:
            q = RootsToPolynomial._func(torch.cat([q, q.conj()], dim=-1))
            a = 0.5 * q[..., 1:-1]
        else:
            p = RootsToPolynomial._func(torch.cat([p, p.conj()], dim=-1))
            q = RootsToPolynomial._func(torch.cat([q, q.conj()], dim=-1))
            p = F.conv1d(p, kernel_p, padding=1 if M % 2 == 1 else 0)
            q = F.conv1d(q, kernel_q)
            a = 0.5 * (p + q)

        a = a.view_as(w)
        a = torch.cat((K, a), dim=-1)
        return a
