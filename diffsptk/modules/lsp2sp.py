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

from ..typing import Callable, Precomputed
from ..utils.private import check_size, filter_values, to
from .base import BaseFunctionalModule

LOG_ZERO = -1.0e10


class LineSpectralPairsToSpectrum(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mglsp2sp.html>`_
    for details.

    Parameters
    ----------
    lsp_order : int >= 0
        The order of the line spectral pairs, :math:`M`.

    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    alpha : float in (-1, 1)
        The warping factor, :math:`\\alpha`.

    gamma : float in [-1, 0)
        The gamma parameter, :math:`\\gamma`.

    log_gain : bool
        If True, assume the input gain is in logarithmic scale.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        The output format.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] A. V. Oppenheim et al., "Discrete representation of signals," *Proceedings of
           the IEEE*, vol. 60, no. 6, pp. 681-691, 1972.

    .. [2] N. Sugamura et al., "Speech data compression by LSP speech analysis-synthesis
           technique," *IEICE trans*, vol. J64-A, no. 8, pp. 599-606, 1981.

    """

    def __init__(
        self,
        lsp_order: int,
        fft_length: int,
        alpha: float = 0,
        gamma: float = -1,
        log_gain: bool = False,
        out_format: str | int = "power",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = lsp_order + 1

        self.values, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("cos_omega", tensors[0])
        self.register_buffer("p_bias", tensors[1])
        self.register_buffer("q_bias", tensors[2])

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Convert line spectral pairs to spectrum.

        Parameters
        ----------
        w : Tensor [shape=(..., M+1)]
            The line spectral pairs in radians.

        Returns
        -------
        out : Tensor [shape=(..., L/2+1)]
            The spectrum.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([ 2.1110, -1.4767,  1.2490,  2.4201,  1.5429])
        >>> lpc = diffsptk.LPC(3, 5)
        >>> a = lpc(x)
        >>> lpc2lsp = diffsptk.LinearPredictiveCoefficientsToLineSpectralPairs(3)
        >>> w = lpc2lsp(a)
        >>> lsp2sp = diffsptk.LineSpectralPairsToSpectrum(3, 8)
        >>> sp = lsp2sp(w)
        >>> sp
        tensor([31.3541, 13.7932, 14.7454, 16.9510, 10.4759])

        """
        check_size(w.size(-1), self.in_dim, "dimension of LSP")
        return self._forward(w, *self.values, **self._buffers)

    @staticmethod
    def _func(w: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, tensors = LineSpectralPairsToSpectrum._precompute(
            w.size(-1) - 1, *args, **kwargs, device=w.device, dtype=w.dtype
        )
        return LineSpectralPairsToSpectrum._forward(w, *values, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(lsp_order: int, fft_length: int, alpha: float, gamma: float) -> None:
        if lsp_order < 0:
            raise ValueError("lsp_order must be non-negative.")
        if fft_length <= 1:
            raise ValueError("fft_length must be greater than 1.")
        if 1 <= abs(alpha):
            raise ValueError("alpha must be in (-1, 1).")
        if not (-1 <= gamma < 0):
            raise ValueError("gamma must be in [-1, 0).")

    @staticmethod
    def _precompute(
        lsp_order: int,
        fft_length: int,
        alpha: float,
        gamma: float,
        log_gain: bool,
        out_format: str | int,
        dtype: torch.dtype | None,
        device: torch.device | None,
    ) -> Precomputed:
        LineSpectralPairsToSpectrum._check(lsp_order, fft_length, alpha, gamma)

        if out_format in (0, "db"):
            formatter = lambda x: x * (20 / np.log(10))
        elif out_format in (1, "log-magnitude"):
            formatter = lambda x: x
        elif out_format in (2, "magnitude"):
            formatter = lambda x: torch.exp(x)
        elif out_format in (3, "power"):
            formatter = lambda x: torch.exp(2 * x)
        else:
            raise ValueError(f"out_format {out_format} is not supported.")

        c1 = 0.5 / gamma
        c2 = np.log(2) * (lsp_order if lsp_order % 2 == 0 else (lsp_order - 1))

        omega = torch.linspace(
            0, torch.pi, fft_length // 2 + 1, device=device, dtype=torch.double
        )
        warped_omega = omega + 2 * torch.atan(
            alpha * torch.sin(omega) / (1 - alpha * torch.cos(omega))
        )
        cos_omega = torch.cos(warped_omega).view(-1, 1)
        cos_omega = to(cos_omega, dtype=dtype)

        def floor_log(x):
            return torch.nan_to_num(torch.log(x), nan=LOG_ZERO, neginf=LOG_ZERO)

        if lsp_order % 2 == 0:
            p = floor_log(torch.sin(0.5 * warped_omega))
            q = floor_log(torch.cos(0.5 * warped_omega))
        else:
            p = floor_log(torch.sin(warped_omega))
            q = torch.zeros_like(warped_omega)
        p_bias = to(p, dtype=dtype)
        q_bias = to(q, dtype=dtype)

        return (log_gain, formatter, c1, c2), None, (cos_omega, p_bias, q_bias)

    @staticmethod
    def _forward(
        w: torch.Tensor,
        log_gain: bool,
        formatter: Callable,
        c1: float,
        c2: float,
        cos_omega: torch.Tensor,
        p_bias: torch.Tensor,
        q_bias: torch.Tensor,
    ) -> torch.Tensor:
        def floor_log(x):
            return torch.clip(torch.log(x), min=LOG_ZERO)

        K, w = torch.split(w, [1, w.size(-1) - 1], dim=-1)
        if not log_gain:
            K = floor_log(K)

        cos_w = torch.cos(w).unsqueeze(-2)
        pq = floor_log(torch.abs(cos_omega - cos_w))  # [..., L/2+1, M]
        p = pq[..., 1::2].sum(-1)
        q = pq[..., 0::2].sum(-1)
        r = torch.logsumexp(2 * torch.stack([p + p_bias, q + q_bias], dim=-1), dim=-1)
        sp = K + c1 * (c2 + r)
        sp = formatter(sp)
        return sp
