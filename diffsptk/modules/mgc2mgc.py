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
import torch.nn.functional as F

from ..misc.utils import cexp
from ..misc.utils import check_size
from ..misc.utils import clog
from ..misc.utils import to
from .freqt import FrequencyTransform
from .gnorm import GeneralizedCepstrumGainNormalization as GainNormalization
from .ignorm import (
    GeneralizedCepstrumInverseGainNormalization as InverseGainNormalization,
)


class GeneralizedCepstrumToGeneralizedCepstrum(nn.Module):
    def __init__(self, in_order, out_order, in_gamma, out_gamma, n_fft=512):
        super().__init__()

        assert 0 <= in_order
        assert 0 <= out_order
        assert abs(in_gamma) <= 1
        assert abs(out_gamma) <= 1
        assert max(in_order, out_order) + 1 < n_fft

        self.in_order = in_order
        self.out_order = out_order
        self.in_gamma = in_gamma
        self.out_gamma = out_gamma
        self.n_fft = n_fft

    def forward(self, c):
        check_size(c.size(-1), self.in_order + 1, "dimension of cepstrum")
        return self._forward(
            c, self.out_order, self.in_gamma, self.out_gamma, self.n_fft
        )

    @staticmethod
    def _forward(c1, out_order, in_gamma, out_gamma, n_fft):
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

        c02 = torch.fft.ifft(C2)[..., : out_order + 1].real
        c2 = torch.cat((c1[..., :1], 2 * c02[..., 1:]), dim=-1)
        return c2

    _func = _forward


class GammaDivision(nn.Module):
    def __init__(self, cep_order, gamma):
        super().__init__()
        g = torch.full((cep_order + 1,), 1 / gamma)
        g[0] = 1
        self.register_buffer("g", to(g))

    def forward(self, c):
        return c * self.g

    @staticmethod
    def _func(c, gamma):
        c0, c1 = torch.split(c, [1, c.size(-1) - 1], dim=-1)
        return torch.cat((c0, c1 / gamma), dim=-1)


class GammaMultiplication(nn.Module):
    def __init__(self, cep_order, gamma):
        super().__init__()
        g = torch.full((cep_order + 1,), gamma)
        g[0] = 1
        self.register_buffer("g", to(g))

    def forward(self, c):
        return c * self.g

    @staticmethod
    def _func(c, gamma):
        c0, c1 = torch.split(c, [1, c.size(-1) - 1], dim=-1)
        return torch.cat((c0, c1 * gamma), dim=-1)


class ZerothGammaDivision(nn.Module):
    def __init__(self, cep_order, gamma):
        super().__init__()
        self.cep_order = cep_order
        self.g = 1 / gamma

    def forward(self, c):
        c0, c1 = torch.split(c, [1, self.cep_order], dim=-1)
        return torch.cat(((c0 - 1) * self.g, c1), dim=-1)

    @staticmethod
    def _func(c, gamma):
        c0, c1 = torch.split(c, [1, c.size(-1) - 1], dim=-1)
        return torch.cat(((c0 - 1) / gamma, c1), dim=-1)


class ZerothGammaMultiplication(nn.Module):
    def __init__(self, cep_order, gamma):
        super().__init__()
        self.cep_order = cep_order
        self.g = gamma

    def forward(self, c):
        c0, c1 = torch.split(c, [1, self.cep_order], dim=-1)
        return torch.cat((c0 * self.g + 1, c1), dim=-1)

    @staticmethod
    def _func(c, gamma):
        c0, c1 = torch.split(c, [1, c.size(-1) - 1], dim=-1)
        return torch.cat((c0 * gamma + 1, c1), dim=-1)


class MelGeneralizedCepstrumToMelGeneralizedCepstrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mgc2mgc.html>`_
    for details. The conversion uses FFT instead of recursive formula.

    Parameters
    ----------
    in_order : int >= 0
        Order of input cepstrum, :math:`M_1`.

    out_order : int >= 0
        Order of output cepstrum, :math:`M_2`.

    in_alpha : float in (-1, 1)
        Input alpha, :math:`\\alpha_1`.

    out_alpha : float in (-1, 1)
        Output alpha, :math:`\\alpha_2`.

    in_gamma : float in [-1, 1]
        Input gamma, :math:`\\gamma_1`.

    out_gamma : float in [-1, 1]
        Output gamma, :math:`\\gamma_2`.

    in_norm : bool
        If True, assume normalized input.

    out_norm : bool
        If True, assume normalized output.

    in_mul : bool
        If True, assume gamma-multiplied input.

    out_mul : bool
        If True, assume gamma-multiplied output.

    n_fft : int >> M1, M2
        Number of FFT bins. Accurate conversion requires the large value.

    References
    ----------
    .. [1] K. Tokuda et al., "Mel-generalized cepstral analysis - A unified approach to
           speech spectral estimation", *Proceedings of ICSLP*, pp. 1043-1046, 1996.

    """

    def __init__(
        self,
        in_order,
        out_order,
        in_alpha=0,
        out_alpha=0,
        in_gamma=0,
        out_gamma=0,
        in_norm=False,
        out_norm=False,
        in_mul=False,
        out_mul=False,
        n_fft=512,
    ):
        super().__init__()

        seq = self._precompute(
            True,
            in_order,
            out_order,
            in_alpha,
            out_alpha,
            in_gamma,
            out_gamma,
            in_norm,
            out_norm,
            in_mul,
            out_mul,
            n_fft,
        )
        self.seq = nn.Sequential(*seq)

    def forward(self, mc):
        """Convert mel-generalized cepstrum to mel-generalized cepstrum.

        Parameters
        ----------
        mc : Tensor [shape=(..., M1+1)]
            Input mel-cepstrum.

        Returns
        -------
        out : Tensor [shape=(..., M2+1)]
            Converted mel-cepstrum.

        Examples
        --------
        >>> c1 = diffsptk.ramp(3)
        >>> mgc2mgc = diffsptk.MelGeneralizedCepstrumToMelGeneralizedCepstrum(3, 4, 0.1)
        >>> c2 = mgc2mgc(c1)
        >>> c2
        tensor([-0.0830,  0.6831,  1.1464,  3.1334,  0.9063])

        """
        return self.seq(mc)

    @staticmethod
    def _func(
        mc,
        out_order,
        in_alpha,
        out_alpha,
        in_gamma,
        out_gamma,
        in_norm,
        out_norm,
        in_mul,
        out_mul,
        n_fft,
    ):
        seq = MelGeneralizedCepstrumToMelGeneralizedCepstrum._precompute(
            False,
            mc.size(-1) - 1,
            out_order,
            in_alpha,
            out_alpha,
            in_gamma,
            out_gamma,
            in_norm,
            out_norm,
            in_mul,
            out_mul,
            n_fft,
        )
        for func in seq:
            mc = func(mc)
        return mc

    @staticmethod
    def _precompute(
        module,
        in_order,
        out_order,
        in_alpha,
        out_alpha,
        in_gamma,
        out_gamma,
        in_norm,
        out_norm,
        in_mul,
        out_mul,
        n_fft,
    ):
        def choice(use_module, module, module_params, common_params):
            if use_module:
                return module(*module_params, *common_params)
            else:
                return lambda c: module._func(c, *common_params)

        assert not (0 == in_gamma and in_mul)

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

        return seq
