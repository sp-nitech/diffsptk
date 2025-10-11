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


class InverseVectorQuantization(BaseNonFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/imsvq.html>`_
    for details.

    References
    ----------
    .. [1] A. v. d. Oord et al., "Neural discrete representation learning," *Advances in
           Neural Information Processing Systems*, pp. 6309-6318, 2017.

    """

    def __init__(self):
        super().__init__()

    def forward(self, indices: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """Perform inverse vector quantization.

        Parameters
        ----------
        indices : Tensor [shape=(...,)]
            The codebook indices.

        codebook : Tensor [shape=(K, M+1)]
            The codebook.

        Returns
        -------
        xq : Tensor [shape=(..., M+1)]
            The quantized vectors.

        Examples
        --------
        >>> import diffsptk
        >>> vq = diffsptk.VectorQuantization(4, 3)
        >>> ivq = diffsptk.InverseVectorQuantization()
        >>> indices = torch.tensor([0, 1, 2, 1])
        >>> xq = ivq(indices, vq.codebook)
        >>> xq.shape
        torch.Size([4, 5])

        """
        target_shape = list(indices.shape)
        target_shape.append(codebook.size(-1))
        xq = torch.index_select(codebook, 0, indices.view(-1).long())
        xq = xq.view(target_shape)
        return xq
