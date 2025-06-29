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

from ..typing import Precomputed
from ..utils.private import filter_values
from .base import BaseFunctionalModule


class SignalToNoiseRatio(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/snr.html>`_
    for details.

    Parameters
    ----------
    frame_length : int >= 1 or None
        The frame length in samples, :math:`L`. If given, calculate the segmental SNR.

    full : bool
        If True, include the constant term in the SNR calculation.

    reduction : ['none', 'mean', 'sum']
        The reduction type.

    eps : float >= 0
        A small value to avoid NaN.

    """

    def __init__(
        self,
        frame_length: int | None = None,
        full: bool = False,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, s: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        """Calculate SNR.

        Parameters
        ----------
        s : Tensor [shape=(..., T)]
            The signal.

        sn : Tensor [shape=(..., T)]
            The signal with noise.

        Returns
        -------
        out : Tensor [shape=(...,) or scalar]
            The SNR.

        Examples
        --------
        >>> s = diffsptk.nrand(4)
        >>> s
        tensor([-0.5804, -0.8002, -0.0645,  0.6101,  0.4396])
        >>> n = diffsptk.nrand(4) * 0.1
        >>> n
        tensor([ 0.0854,  0.0485, -0.0826,  0.1455,  0.0257])
        >>> snr = diffsptk.SignalToNoiseRatio(full=True)
        >>> y = snr(s, s + n)
        >>> y
        tensor(16.0614)

        """
        return self._forward(s, sn, *self.values)

    @staticmethod
    def _func(s: torch.Tensor, sn: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = SignalToNoiseRatio._precompute(*args, **kwargs)
        return SignalToNoiseRatio._forward(s, sn, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(frame_length: int | None, eps: float) -> None:
        if frame_length is not None and frame_length <= 0:
            raise ValueError("frame_length must be positive.")
        if eps < 0:
            raise ValueError("eps must be non-negative.")

    @staticmethod
    def _precompute(
        frame_length: int | None, full: bool, reduction: str, eps: float
    ) -> Precomputed:
        SignalToNoiseRatio._check(frame_length, eps)
        const = 10 if full else 1
        return (frame_length, reduction, eps, const)

    @staticmethod
    def _forward(
        s: torch.Tensor,
        sn: torch.Tensor,
        frame_length: int | None,
        reduction: str,
        eps: float,
        const: float,
    ) -> torch.Tensor:
        if frame_length is not None:
            s = s.unfold(-1, frame_length, frame_length)
            sn = sn.unfold(-1, frame_length, frame_length)

        s2 = torch.square(s).sum(-1)
        n2 = torch.square(sn - s).sum(-1)
        snr = torch.log10((s2 + eps) / (n2 + eps))
        if frame_length is not None:
            snr = snr.squeeze(-1)

        if reduction == "none":
            pass
        elif reduction == "sum":
            snr = snr.sum()
        elif reduction == "mean":
            snr = snr.mean()
        else:
            raise ValueError(f"reduction {reduction} is not supported.")

        return const * snr
