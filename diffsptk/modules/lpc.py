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
from .acorr import Autocorrelation
from .base import BaseFunctionalModule
from .levdur import LevinsonDurbin


class LinearPredictiveCodingAnalysis(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/lpc.html>`_
    for details. Double precision is recommended.

    Parameters
    ----------
    frame_length : int > M
        The frame length, :math:`L`.

    lpc_order : int >= 0
        The order of the LPC coefficients, :math:`M`.

    eps : float >= 0
        A small value to improve numerical stability.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        frame_length: int,
        lpc_order: int,
        eps: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        _, layers, _ = self._precompute(**filter_values(locals()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform LPC analysis.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            The framed waveform.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The gain and LPC coefficients.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        tensor([ 0.8226, -0.0284, -0.5715,  0.2127,  0.1217])
        >>> lpc = diffsptk.LPC(5, 2)
        >>> a = lpc(x)
        >>> a
        tensor([0.8726, 0.1475, 0.5270])

        """
        return self._forward(x, *self.layers)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, layers, _ = LinearPredictiveCodingAnalysis._precompute(
            x.size(-1), *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return LinearPredictiveCodingAnalysis._forward(x, *layers)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check() -> None:
        pass

    @staticmethod
    def _precompute(
        frame_length: int,
        lpc_order: int,
        eps: float,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        LinearPredictiveCodingAnalysis._check()
        module = inspect.stack()[1].function != "_func"

        acorr = get_layer(
            module,
            Autocorrelation,
            dict(
                frame_length=frame_length,
                acr_order=lpc_order,
            ),
        )
        levdur = get_layer(
            module,
            LevinsonDurbin,
            dict(
                lpc_order=lpc_order,
                eps=eps,
                device=device,
                dtype=dtype,
            ),
        )
        return None, (acorr, levdur), None

    @staticmethod
    def _forward(x: torch.Tensor, acorr: Callable, levdur: Callable) -> torch.Tensor:
        return levdur(acorr(x))
