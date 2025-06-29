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

import inspect

import torch
import torch.nn.functional as F
from torch import nn

from ..typing import Callable, Precomputed
from ..utils.private import cexp, check_size, clog, filter_values, to
from .base import BaseFunctionalModule
from .freqt import FrequencyTransform
from .gnorm import GeneralizedCepstrumGainNormalization as GainNormalization
from .ignorm import (
    GeneralizedCepstrumInverseGainNormalization as InverseGainNormalization,
)


class MelGeneralizedCepstrumToMelGeneralizedCepstrum(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mgc2mgc.html>`_
    for details.

    Parameters
    ----------
    in_order : int >= 0
        The order of the input cepstrum, :math:`M_1`.

    out_order : int >= 0
        The order of the output cepstrum, :math:`M_2`.

    in_alpha : float in (-1, 1)
        The input alpha, :math:`\\alpha_1`.

    out_alpha : float in (-1, 1)
        The output alpha, :math:`\\alpha_2`.

    in_gamma : float in [-1, 1]
        The input gamma, :math:`\\gamma_1`.

    out_gamma : float in [-1, 1]
        The output gamma, :math:`\\gamma_2`.

    in_norm : bool
        If True, the input is assumed to be normalized.

    out_norm : bool
        If True, the output is assumed to be normalized.

    in_mul : bool
        If True, the input is assumed to be gamma-multiplied.

    out_mul : bool
        If True, the output is assumed to be gamma-multiplied.

    n_fft : int >> M1, M2
        The number of FFT bins. Accurate conversion requires a large value.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] K. Tokuda et al., "Mel-generalized cepstral analysis - A unified approach to
           speech spectral estimation", *Proceedings of ICSLP*, pp. 1043-1046, 1996.

    """

    def __init__(
        self,
        in_order: int,
        out_order: int,
        in_alpha: float = 0,
        out_alpha: float = 0,
        in_gamma: float = 0,
        out_gamma: float = 0,
        in_norm: bool = False,
        out_norm: bool = False,
        in_mul: bool = False,
        out_mul: bool = False,
        n_fft: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        _, layers, _ = self._precompute(**filter_values(locals()))
        self.seq = nn.Sequential(*layers[0])

    def forward(self, mc: torch.Tensor) -> torch.Tensor:
        """Convert mel-generalized cepstrum to mel-generalized cepstrum.

        Parameters
        ----------
        mc : Tensor [shape=(..., M1+1)]
            The input mel-cepstrum.

        Returns
        -------
        out : Tensor [shape=(..., M2+1)]
            The output mel-cepstrum.

        Examples
        --------
        >>> c1 = diffsptk.ramp(3)
        >>> mgc2mgc = diffsptk.MelGeneralizedCepstrumToMelGeneralizedCepstrum(3, 4, 0.1)
        >>> c2 = mgc2mgc(c1)
        >>> c2
        tensor([-0.0830,  0.6831,  1.1464,  3.1334,  0.9063])

        """
        return self._forward(mc, self.seq)

    @staticmethod
    def _func(mc: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, layers, _ = MelGeneralizedCepstrumToMelGeneralizedCepstrum._precompute(
            mc.size(-1) - 1, *args, **kwargs, device=mc.device, dtype=mc.dtype
        )

        def seq(x):
            for layer in layers[0]:
                x = layer(x)
            return x

        return MelGeneralizedCepstrumToMelGeneralizedCepstrum._forward(mc, seq)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(
        in_order: int,
        out_order: int,
        in_alpha: float,
        out_alpha: float,
        in_gamma: float,
        out_gamma: float,
        in_mul: bool,
        n_fft: int,
    ) -> None:
        if in_order < 0:
            raise ValueError("in_order must be non-negative.")
        if out_order < 0:
            raise ValueError("out_order must be non-negative.")
        if 1 <= abs(in_alpha):
            raise ValueError("in_alpha must be in (-1, 1).")
        if 1 <= abs(out_alpha):
            raise ValueError("out_alpha must be in (-1, 1).")
        if 1 < abs(in_gamma):
            raise ValueError("in_gamma must be in [-1, 1].")
        if 1 < abs(out_gamma):
            raise ValueError("out_gamma must be in [-1, 1].")
        if n_fft <= max(in_order, out_order) + 1:
            raise ValueError("n_fft must be much larger then order of cepstrum.")
        if 0 == in_gamma and in_mul:
            raise ValueError("Invalid combination of in_gamma and in_mul.")

    @staticmethod
    def _precompute(
        in_order: int,
        out_order: int,
        in_alpha: float,
        out_alpha: float,
        in_gamma: float,
        out_gamma: float,
        in_norm: bool,
        out_norm: bool,
        in_mul: bool,
        out_mul: bool,
        n_fft: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        def choice(use_module, module, module_params, common_params):
            other_params = {}
            if "device" in inspect.signature(module.__init__).parameters:
                other_params["device"] = device
            if "dtype" in inspect.signature(module.__init__).parameters:
                other_params["dtype"] = dtype
            if use_module:
                return module(*module_params, *common_params, **other_params)
            else:
                return lambda c: module._func(c, *common_params)

        MelGeneralizedCepstrumToMelGeneralizedCepstrum._check(
            in_order, out_order, in_alpha, out_alpha, in_gamma, out_gamma, in_mul, n_fft
        )
        module = inspect.stack()[1].function != "_func"

        seq = []
        if not in_norm and in_mul:
            seq.append(choice(module, ZerothGammaDivision, [in_order], [in_gamma]))

        alpha = (out_alpha - in_alpha) / (1 - in_alpha * out_alpha)
        if 0 == alpha:
            if in_order == out_order and in_gamma == out_gamma:
                if not in_mul and out_mul:
                    seq.append(
                        choice(module, GammaMultiplication, [in_order], [in_gamma])
                    )
                if not in_norm and out_norm:
                    seq.append(
                        choice(module, GainNormalization, [in_order], [in_gamma])
                    )
                if in_norm and not out_norm:
                    seq.append(
                        choice(
                            module, InverseGainNormalization, [out_order], [out_gamma]
                        )
                    )
                if in_mul and not out_mul:
                    seq.append(choice(module, GammaDivision, [out_order], [out_gamma]))
            else:
                if in_mul:
                    seq.append(choice(module, GammaDivision, [in_order], [in_gamma]))
                if not in_norm:
                    seq.append(
                        choice(module, GainNormalization, [in_order], [in_gamma])
                    )
                if True:
                    seq.append(
                        choice(
                            module,
                            GeneralizedCepstrumToGeneralizedCepstrum,
                            [in_order],
                            [out_order, in_gamma, out_gamma, n_fft],
                        )
                    )
                if not out_norm:
                    seq.append(
                        choice(
                            module, InverseGainNormalization, [out_order], [out_gamma]
                        )
                    )
                if out_mul:
                    seq.append(
                        choice(module, GammaMultiplication, [out_order], [out_gamma])
                    )
        else:
            if in_mul:
                seq.append(choice(module, GammaDivision, [in_order], [in_gamma]))
            if in_norm:
                seq.append(
                    choice(module, InverseGainNormalization, [in_order], [in_gamma])
                )
            if True:
                seq.append(
                    choice(module, FrequencyTransform, [in_order], [out_order, alpha])
                )
            if out_norm or in_gamma != out_gamma:
                seq.append(choice(module, GainNormalization, [out_order], [in_gamma]))
            if in_gamma != out_gamma:
                seq.append(
                    choice(
                        module,
                        GeneralizedCepstrumToGeneralizedCepstrum,
                        [out_order],
                        [out_order, in_gamma, out_gamma, n_fft],
                    )
                )
            if not out_norm and in_gamma != out_gamma:
                seq.append(
                    choice(module, InverseGainNormalization, [out_order], [out_gamma])
                )
            if out_mul:
                seq.append(
                    choice(module, GammaMultiplication, [out_order], [out_gamma])
                )

        if not out_norm and out_mul:
            seq.append(
                choice(module, ZerothGammaMultiplication, [out_order], [out_gamma])
            )

        return None, (seq,), None

    @staticmethod
    def _forward(mc: torch.Tensor, seq: Callable) -> torch.Tensor:
        return seq(mc)


class GeneralizedCepstrumToGeneralizedCepstrum(nn.Module):
    def __init__(
        self,
        in_order: int,
        out_order: int,
        in_gamma: float,
        out_gamma: float,
        n_fft: int = 512,
    ) -> None:
        super().__init__()

        self.in_order = in_order
        self.out_order = out_order
        self.in_gamma = in_gamma
        self.out_gamma = out_gamma
        self.n_fft = n_fft

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        check_size(c.size(-1), self.in_order + 1, "dimension of cepstrum")
        return self._forward(
            c, self.out_order, self.in_gamma, self.out_gamma, self.n_fft
        )

    @staticmethod
    def _forward(
        c1: torch.Tensor,
        out_order: int,
        in_gamma: float,
        out_gamma: float,
        n_fft: int,
    ) -> torch.Tensor:
        c01 = F.pad(c1[..., 1:], (1, 0))
        C1 = torch.fft.fft(c01, n=n_fft)

        if in_gamma == 0:
            sC1 = cexp(C1)
        else:
            C1 *= in_gamma
            C1.real += 1
            r = C1.abs() ** (1 / in_gamma)
            theta = C1.angle() / in_gamma
            sC1 = torch.polar(r, theta)

        if out_gamma == 0:
            C2 = clog(sC1)
        else:
            r = sC1.abs() ** out_gamma
            theta = sC1.angle() * out_gamma
            C2 = (r * torch.cos(theta) - 1) / out_gamma

        c02 = torch.fft.ifft(C2).real[..., : out_order + 1]
        c2 = torch.cat((c1[..., :1], 2 * c02[..., 1:]), dim=-1)
        return c2

    _func = _forward


class GammaDivision(nn.Module):
    def __init__(
        self,
        cep_order: int,
        gamma: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        g = torch.full((cep_order + 1,), 1 / gamma, device=device, dtype=torch.double)
        g[0] = 1
        self.register_buffer("g", to(g, dtype=dtype))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return c * self.g

    @staticmethod
    def _func(c: torch.Tensor, gamma: float) -> torch.Tensor:
        c0, c1 = torch.split(c, [1, c.size(-1) - 1], dim=-1)
        return torch.cat((c0, c1 / gamma), dim=-1)


class GammaMultiplication(nn.Module):
    def __init__(
        self,
        cep_order: int,
        gamma: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        g = torch.full((cep_order + 1,), gamma, device=device, dtype=torch.double)
        g[0] = 1
        self.register_buffer("g", to(g, dtype=dtype))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return c * self.g

    @staticmethod
    def _func(c: torch.Tensor, gamma: float) -> torch.Tensor:
        c0, c1 = torch.split(c, [1, c.size(-1) - 1], dim=-1)
        return torch.cat((c0, c1 * gamma), dim=-1)


class ZerothGammaDivision(nn.Module):
    def __init__(self, cep_order: int, gamma: float) -> None:
        super().__init__()
        self.cep_order = cep_order
        self.g = 1 / gamma

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        c0, c1 = torch.split(c, [1, self.cep_order], dim=-1)
        return torch.cat(((c0 - 1) * self.g, c1), dim=-1)

    @staticmethod
    def _func(c: torch.Tensor, gamma: float) -> torch.Tensor:
        c0, c1 = torch.split(c, [1, c.size(-1) - 1], dim=-1)
        return torch.cat(((c0 - 1) / gamma, c1), dim=-1)


class ZerothGammaMultiplication(nn.Module):
    def __init__(self, cep_order: int, gamma: float) -> None:
        super().__init__()
        self.cep_order = cep_order
        self.g = gamma

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        c0, c1 = torch.split(c, [1, self.cep_order], dim=-1)
        return torch.cat((c0 * self.g + 1, c1), dim=-1)

    @staticmethod
    def _func(c: torch.Tensor, gamma: float) -> torch.Tensor:
        c0, c1 = torch.split(c, [1, c.size(-1) - 1], dim=-1)
        return torch.cat((c0 * gamma + 1, c1), dim=-1)
