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
from ..utils.private import UNVOICED_SYMBOL, filter_values
from .base import BaseFunctionalModule
from .rmse import RootMeanSquareError


class F0Evaluation(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/f0eval.html>`_
    for details. Note that the gradients cannot be calculated if the output format
    is related to voiced/unvoiced decision.

    Parameters
    ----------
    reduction : ['none', 'mean', 'sum']
        The reduction type.

    out_format : ['f0-rmse-hz', 'f0-rmse-cent', 'f0-rmse-semitone', 'vuv-error-rate', \
                  'vuv-error-percent', 'vuv-macro-f1-score']
        The output format.

    """

    def __init__(
        self, reduction: str = "mean", out_format: str = "f0-rmse-cent"
    ) -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate F0 metric.

        Parameters
        ----------
        x : Tensor [shape=(..., N)]
            The input F0 in Hz.

        y : Tensor [shape=(..., N)]
            The target F0 in Hz.

        Returns
        -------
        out : Tensor [shape=(...,) or scalar]
            The F0 metric.

        """
        return self._forward(x, y, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = F0Evaluation._precompute(*args, **kwargs)
        return F0Evaluation._forward(x, y, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check() -> None:
        pass

    @staticmethod
    def _precompute(reduction: str, out_format: str) -> Precomputed:
        F0Evaluation._check()
        return (reduction, out_format)

    @staticmethod
    def _forward(
        x: torch.Tensor, y: torch.Tensor, reduction: str, out_format: str
    ) -> torch.Tensor:
        if out_format.startswith("f0-rmse"):
            voiced = (x != UNVOICED_SYMBOL) & (y != UNVOICED_SYMBOL)
            if out_format == "f0-rmse-hz":
                convert = lambda x: x
            elif out_format == "f0-rmse-cent":
                convert = lambda x: 1200 * torch.log2(x)
            elif out_format == "f0-rmse-semitone":
                convert = lambda x: 12 * torch.log2(x)
            else:
                raise ValueError(f"out_format {out_format} is not supported.")
            out = RootMeanSquareError._func(
                convert(x[voiced]), convert(y[voiced]), "none"
            )
        else:
            TP = torch.sum((x != UNVOICED_SYMBOL) & (y != UNVOICED_SYMBOL), dim=-1)
            FP = torch.sum((x == UNVOICED_SYMBOL) & (y != UNVOICED_SYMBOL), dim=-1)
            FN = torch.sum((x != UNVOICED_SYMBOL) & (y == UNVOICED_SYMBOL), dim=-1)
            TN = torch.sum((x == UNVOICED_SYMBOL) & (y == UNVOICED_SYMBOL), dim=-1)
            FPFN = FP + FN
            if out_format == "vuv-error-rate":
                out = FPFN / x.shape[-1]
            elif out_format == "vuv-error-percent":
                out = 100 * FPFN / x.shape[-1]
            elif out_format == "vuv-macro-f1-score":
                f1_score_pos = torch.nan_to_num((2 * TP) / (2 * TP + FPFN))
                f1_score_neg = torch.nan_to_num((2 * TN) / (2 * TN + FPFN))
                out = (f1_score_pos + f1_score_neg) / 2
            else:
                raise ValueError(f"out_format {out_format} is not supported.")

        if reduction == "none":
            pass
        elif reduction == "sum":
            out = out.sum()
        elif reduction == "mean":
            out = out.mean()
        else:
            raise ValueError(f"reduction {reduction} is not supported.")

        return out
