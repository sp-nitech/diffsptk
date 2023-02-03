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
import torch.nn as nn


class InverseMultiStageVectorQuantization(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/imsvq.html>`_
    for details.

    """

    def __init__(self):
        super(InverseMultiStageVectorQuantization, self).__init__()

    def forward(self, indices, codebooks):
        """Perform inverse residual vector quantization.

        Parameters
        ----------
        indices : Tensor [shape=(..., Q)]
            Codebook indices.

        codebooks : Tensor [shape=(Q, K, M+1)]
            Codebooks.

        Returns
        -------
        xq : Tensor [shape=(..., M+1)]
            Quantized vectors.

        Examples
        --------
        >>> msvq = diffsptk.MultiStageVectorQuantization(4, 3, 2)
        >>> imsvq = diffsptk.InverseMultiStageVectorQuantization()
        >>> indices = torch.tensor([[0, 1], [1, 0]])
        >>> xq = imsvq(indices, msvq.codebooks)
        >>> xq
        tensor([[-0.8029, -0.1674,  0.5697,  0.9734,  0.1920],
                [ 0.0720, -1.0491, -0.4491, -0.2043, -0.3582]])

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
