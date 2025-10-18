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

from .base import BaseNonFunctionalModule


class InverseMultiStageVectorQuantization(BaseNonFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/imsvq.html>`_
    for details.

    References
    ----------
    .. [1] A. v. d. Oord et al., "Neural discrete representation learning," *Advances in
           Neural Information Processing Systems*, pp. 6309-6318, 2017.

    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, indices: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
        """Perform inverse residual vector quantization.

        Parameters
        ----------
        indices : Tensor [shape=(..., Q)]
            The codebook indices.

        codebooks : Tensor [shape=(Q, K, M+1)]
            The codebooks.

        Returns
        -------
        xq : Tensor [shape=(..., M+1)]
            The quantized vectors.

        Examples
        --------
        >>> import diffsptk
        >>> msvq = diffsptk.MultiStageVectorQuantization(4, 3, n_stage=2)
        >>> imsvq = diffsptk.InverseMultiStageVectorQuantization()
        >>> indices = torch.tensor([[0, 1], [2, 1]])
        >>> xq = imsvq(indices, msvq.codebooks)
        >>> xq.shape
        torch.Size([2, 5])

        """
        target_shape = list(indices.shape[:-1])
        target_shape.append(codebooks.size(-1))
        xq = 0
        for i in range(indices.size(-1)):
            code_vector = torch.index_select(
                codebooks[i], 0, indices[..., i].view(-1).long()
            )
            xq = xq + code_vector
        xq = xq.view(target_shape)
        return xq
