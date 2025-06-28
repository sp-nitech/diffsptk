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
from ..utils.private import TAU, check_size, deconv1d, filter_values
from .base import BaseFunctionalModule
from .root_pol import PolynomialToRoots


class LinearPredictiveCoefficientsToLineSpectralPairs(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lpc2lsp.html>`_
    for details.

    Parameters
    ----------
    lpc_order : int >= 0
        The order of the input LPC, :math:`M`.

    log_gain : bool
        If True, output the gain in logarithmic scale.

    sample_rate : int >= 1 or None
        The sample rate in Hz.

    out_format : ['radian', 'cycle', 'khz', 'hz']
        The output format.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] P. Kabal et al., "The computation of line spectral frequencies using
           Chebyshev polynomials," *IEEE Transactions on Acoustics, Speech, and Signal
           Processing*, vol. 34, no. 6, pp. 1419-1426, 1986.

    """

    def __init__(
        self,
        lpc_order: int,
        log_gain: bool = False,
        sample_rate: int | None = None,
        out_format: str | int = "radian",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = lpc_order + 1

        self.values, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("kernel_p", tensors[0])
        self.register_buffer("kernel_q", tensors[1])

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """Convert LPC to LSP.

        Parameters
        ----------
        a : Tensor [shape=(..., M+1)]
            The LPC coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The LSP frequencies.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([-1.5326,  1.0875, -1.5925,  0.6913,  1.6217])
        >>> lpc = diffsptk.LPC(3, 5)
        >>> a = lpc(x)
        >>> a
        tensor([ 2.7969,  0.3908,  0.0458, -0.0859])
        >>> lpc2lsp = diffsptk.LinearPredictiveCoefficientsToLineSpectralPairs(3)
        >>> w = lpc2lsp(a)
        >>> w
        tensor([2.7969, 0.9037, 1.8114, 2.4514])

        """
        check_size(a.size(-1), self.in_dim, "dimension of LPC")
        return self._forward(a, *self.values, **self._buffers)

    @staticmethod
    def _func(a: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, tensors = (
            LinearPredictiveCoefficientsToLineSpectralPairs._precompute(
                a.size(-1) - 1, *args, **kwargs, device=a.device, dtype=a.dtype
            )
        )
        return LinearPredictiveCoefficientsToLineSpectralPairs._forward(
            a, *values, *tensors
        )

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(
        lpc_order: int, log_gain: bool, sample_rate: int, out_format: str | int
    ) -> None:
        if lpc_order < 0:
            raise ValueError("lpc_order must be non-negative.")
        if out_format in (2, 3, "hz", "khz") and (
            sample_rate is None or sample_rate <= 0
        ):
            raise ValueError("sample_rate must be positive.")

    @staticmethod
    def _precompute(
        lpc_order: int,
        log_gain: bool,
        sample_rate: int,
        out_format: str | int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        LinearPredictiveCoefficientsToLineSpectralPairs._check(
            lpc_order, log_gain, sample_rate, out_format
        )

        if out_format in (0, "radian"):
            formatter = lambda x: x
        elif out_format in (1, "cycle"):
            formatter = lambda x: x / TAU
        elif out_format in (2, "khz"):
            formatter = lambda x: x / (TAU / sample_rate * 1000)
        elif out_format in (3, "hz"):
            formatter = lambda x: x / (TAU / sample_rate)
        else:
            raise ValueError(f"out_format {out_format} is not supported.")

        params = {"device": device, "dtype": dtype}
        if lpc_order % 2 == 0:
            kernel_p = torch.tensor([1.0, -1.0], **params)
            kernel_q = torch.tensor([1.0, 1.0], **params)
        else:
            kernel_p = torch.tensor([1.0, 0.0, -1.0], **params)
            kernel_q = torch.tensor([1.0], **params)

        return (log_gain, formatter), None, (kernel_p, kernel_q)

    @staticmethod
    def _forward(
        a: torch.Tensor,
        log_gain: bool,
        formatter: Callable,
        kernel_p: torch.Tensor,
        kernel_q: torch.Tensor,
    ) -> torch.Tensor:
        M = a.size(-1) - 1
        K, a = torch.split(a, [1, M], dim=-1)
        if log_gain:
            K = torch.log(K)

        if M == 0:
            return K

        a0 = F.pad(a, (1, 0), value=1)
        a1 = F.pad(a0, (0, 1), value=0)
        a2 = a1.flip(-1)
        p = a1 - a2
        q = a1 + a2
        if M == 1:
            q = PolynomialToRoots._func(q)
            w = torch.angle(q[..., 0])
        else:
            p = deconv1d(p, kernel_p)
            q = deconv1d(q, kernel_q)
            p = PolynomialToRoots._func(p)
            q = PolynomialToRoots._func(q)
            p = torch.angle(p[..., 0::2])
            q = torch.angle(q[..., 0::2])
            w, _ = torch.sort(torch.cat((p, q), dim=-1))

        w = w.view_as(a)
        w = formatter(w)
        w = torch.cat((K, w), dim=-1)
        return w
