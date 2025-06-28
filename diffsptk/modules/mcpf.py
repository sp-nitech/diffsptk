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
from torch import nn

from ..typing import Callable, Precomputed
from ..utils.private import filter_values, get_layer
from .b2mc import MLSADigitalFilterCoefficientsToMelCepstrum
from .base import BaseFunctionalModule
from .c2acr import CepstrumToAutocorrelation
from .freqt import FrequencyTransform
from .mc2b import MelCepstrumToMLSADigitalFilterCoefficients


class MelCepstrumPostfiltering(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mcpf.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the mel-cepstrum, :math:`M`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    beta : float
        The intensity parameter, :math:`\\beta`.

    onset : int >= 0
        The onset index.

    ir_length : int >= 1
        The length of the impulse response.

    References
    ----------
    .. [1] T. Yoshimura et al., "Incorporating a mixed excitation model and postfilter
           into HMM-based text-to-speech synthesis," *Systems and Computers in Japan*,
           vol. 36, no. 12, pp. 43-50, 2005.

    """

    def __init__(
        self,
        cep_order: int,
        alpha: float = 0,
        beta: float = 0,
        onset: int = 2,
        ir_length: int = 128,
    ) -> None:
        super().__init__()

        _, layers, tensors = self._precompute(**filter_values(locals()))
        self.layers = nn.ModuleList(layers)
        self.register_buffer("weight", tensors[0])

    def forward(self, mc: torch.Tensor) -> torch.Tensor:
        """Perform mel-cesptrum postfiltering.

        Parameters
        ----------
        mc : Tensor [shape=(..., M+1)]
            The input mel-cepstral coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The postfiltered mel-cepstral coefficients.

        Examples
        --------
        >>> X = diffsptk.nrand(4).square()
        >>> X
        tensor([0.2725, 2.5650, 0.3552, 0.3757, 0.1904])
        >>> mcep = diffsptk.MelCepstralAnalysis(3, 8, 0.1)
        >>> mcpf = diffsptk.MelCepstrumPostfiltering(3, 0.1, 0.2)
        >>> mc1 = mcep(X)
        >>> mc1
        tensor([-0.2819,  0.3486, -0.2487, -0.3600])
        >>> mc2 = mcpf(mc1)
        >>> mc2
        tensor([-0.3256,  0.3486, -0.2984, -0.4320])

        """
        return self._forward(mc, *self.layers, **self._buffers)

    @staticmethod
    def _func(mc: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, layers, tensors = MelCepstrumPostfiltering._precompute(
            mc.size(-1) - 1, *args, **kwargs, device=mc.device, dtype=mc.dtype
        )
        return MelCepstrumPostfiltering._forward(mc, *layers, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(onset: int) -> None:
        if onset < 0:
            raise ValueError("onset must be non-negative.")

    @staticmethod
    def _precompute(
        cep_order: int,
        alpha: float,
        beta: float,
        onset: int,
        ir_length: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        MelCepstrumPostfiltering._check(onset)
        module = inspect.stack()[1].function != "_func"

        freqt = get_layer(
            module,
            FrequencyTransform,
            dict(
                in_order=cep_order,
                out_order=ir_length - 1,
                alpha=-alpha,
            ),
        )
        c2acr = get_layer(
            module,
            CepstrumToAutocorrelation,
            dict(
                cep_order=ir_length - 1,
                acr_order=0,
                n_fft=ir_length,
            ),
        )
        mc2b = get_layer(
            module,
            MelCepstrumToMLSADigitalFilterCoefficients,
            dict(cep_order=cep_order, alpha=alpha),
        )
        b2mc = get_layer(
            module,
            MLSADigitalFilterCoefficientsToMelCepstrum,
            dict(cep_order=cep_order, alpha=alpha),
        )

        weight = torch.full((cep_order + 1,), 1 + beta, device=device, dtype=dtype)
        weight[:onset] = 1
        return None, (freqt, c2acr, mc2b, b2mc), (weight,)

    @staticmethod
    def _forward(
        mc: torch.Tensor,
        freqt: Callable,
        c2acr: Callable,
        mc2b: Callable,
        b2mc: Callable,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        mc1 = mc
        e1 = c2acr(freqt(mc1))

        mc2 = mc * weight
        e2 = c2acr(freqt(mc2))

        b2 = mc2b(mc2)
        b2[..., :1] += 0.5 * torch.log(e1 / e2)
        mc2 = b2mc(b2)
        return mc2
