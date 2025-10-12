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

import warnings

import torch

from ..typing import Precomputed
from ..utils.private import check_size, filter_values, to
from .base import BaseFunctionalModule


class MLSADigitalFilterStabilityCheck(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mlsacheck.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the mel-cepstrum, :math:`M`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    pade_order : int in [4, 7]
        The order of the Pade approximation.

    strict : bool
        If True, prioritizes maintaining the maximum log approximation error over MLSA
        filter stability.

    threshold : float > 0 or None
        The threshold value. If None, it is automatically computed.

    fast : bool
        Enables fast mode (do not use FFT).

    n_fft : int > M
        The number of FFT bins. Used only in non-fast mode.

    warn_type : ['ignore', 'warn', 'exit']
        The warning type.

    mod_type : ['clip', 'scale']
        The modification method.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] S. Imai et al., "Mel log spectrum approximation (MLSA) filter for speech
           synthesis," *Electronics and Communications in Japan*, vol. 66, no. 2,
           pp. 11-18, 1983.

    """

    def __init__(
        self,
        cep_order: int,
        *,
        alpha: float = 0,
        pade_order: int = 4,
        strict: bool = True,
        threshold: float | None = None,
        fast: bool = True,
        n_fft: int = 256,
        warn_type: str = "warn",
        mod_type: str = "scale",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = cep_order + 1

        self.values, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("alpha_vector", tensors[0])

    def forward(self, mc: torch.Tensor) -> torch.Tensor:
        """Check the stability of the MLSA digital filter.

        Parameters
        ----------
        mc : Tensor [shape=(..., M+1)]
            The input mel-cepstrum.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The modified mel-cepstrum.

        Examples
        --------
        >>> import diffsptk
        >>> import torch
        >>> mlsacheck = diffsptk.MLSADigitalFilterStabilityCheck(
        ...     cep_odder=4, alpha=0.1, warn_type="ignore"
        ... )
        >>> c1 = torch.tensor([1.8963, 7.6629, 4.4804, 8.0669, -1.2768])
        >>> c2 = mlsacheck(c1)
        >>> c2
        tensor([ 1.3336,  1.7537,  1.0254,  1.8462, -0.2922])

        """
        check_size(mc.size(-1), self.in_dim, "dimension of mel-cepstrum")
        return self._forward(mc, *self.values, **self._buffers)

    @staticmethod
    def _func(mc: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, tensors = MLSADigitalFilterStabilityCheck._precompute(
            mc.size(-1) - 1, *args, **kwargs, device=mc.device, dtype=mc.dtype
        )
        return MLSADigitalFilterStabilityCheck._forward(mc, *values, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(cep_order: int) -> None:
        if cep_order < 0:
            raise ValueError("cep_order must be non-negative.")

    @staticmethod
    def _precompute(
        cep_order: int,
        alpha: float,
        pade_order: int,
        strict: bool,
        threshold: float | None,
        fast: bool,
        n_fft: int,
        warn_type: str,
        mod_type: str,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        MLSADigitalFilterStabilityCheck._check(cep_order)

        if threshold is None:
            if pade_order == 4:
                threshold = 4.5 if strict else 6.20
            elif pade_order == 5:
                threshold = 6.0 if strict else 7.65
            elif pade_order == 6:
                threshold = 7.4 if strict else 9.13
            elif pade_order == 7:
                threshold = 8.9 if strict else 10.6
            else:
                raise ValueError(f"pade_order {pade_order} is not supported.")

        alpha_vector = (-alpha) ** torch.arange(
            cep_order + 1, device=device, dtype=torch.double
        )

        return (
            (threshold, fast, n_fft, warn_type, mod_type),
            None,
            (to(alpha_vector, dtype=dtype),),
        )

    @staticmethod
    def _forward(
        mc: torch.Tensor,
        threshold: float,
        fast: bool,
        n_fft: int,
        warn_type: str,
        mod_type: str,
        alpha_vector: torch.Tensor,
    ) -> torch.Tensor:
        gain = (mc * alpha_vector).sum(-1, keepdim=True)

        if fast:
            max_amplitude = mc.sum(-1, keepdim=True) - gain
        else:
            c1 = torch.cat((mc[..., :1] - gain, mc[..., 1:]), dim=-1)
            C1 = torch.fft.rfft(c1, n=n_fft)
            C1_amplitude = C1.abs()
            max_amplitude = torch.amax(C1_amplitude, dim=-1, keepdim=True)
        max_amplitude = torch.clip(max_amplitude, min=1e-16)

        if torch.any(threshold < max_amplitude):
            if warn_type == "ignore":
                pass
            elif warn_type == "warn":
                warnings.warn("Detected unstable MLSA filter.")
            elif warn_type == "exit":
                raise RuntimeError("Detected unstable MLSA filter.")
            else:
                raise RuntimeError

        if mod_type == "clip":
            scale = threshold / C1_amplitude
        elif mod_type == "scale":
            scale = threshold / max_amplitude
        else:
            raise ValueError(f"mod_type {mod_type} is not supported.")
        scale = torch.clip(scale, max=1)

        if fast:
            c0, c1 = torch.split(mc, [1, mc.size(-1) - 1], dim=-1)
            c0 = (c0 - gain) * scale + gain
            c1 = c1 * scale
            c2 = torch.cat((c0, c1), dim=-1)
        else:
            c2 = torch.fft.irfft(C1 * scale)[..., : mc.size(-1)]
            c2 = torch.cat((c2[..., :1] + gain, c2[..., 1:]), dim=-1)
        return c2
