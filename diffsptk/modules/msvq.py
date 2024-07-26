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

from torch import nn
from vector_quantize_pytorch import ResidualVQ


class MultiStageVectorQuantization(nn.Module):
    """See `this page <https://github.com/lucidrains/vector-quantize-pytorch>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        Order of vector, :math:`M`.

    codebook_size : int >= 1
        Codebook size, :math:`K`.

    n_stage : int >= 1
        Number of stages (quantizers), :math:`Q`.

    **kwargs : additional keyword arguments
        See `this page`_ for details.

    """

    def __init__(self, order, codebook_size, n_stage, **kwargs):
        super().__init__()

        assert 0 <= order
        assert 1 <= codebook_size
        assert 1 <= n_stage

        self.vq = ResidualVQ(
            dim=order + 1, codebook_size=codebook_size, num_quantizers=n_stage, **kwargs
        ).float()

    @property
    def codebooks(self):
        return self.vq.codebooks

    def forward(self, x, codebooks=None, **kwargs):
        """Perform residual vector quantization.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            Input vectors.

        codebooks : Tensor [shape=(Q, K, M+1)]
            External codebooks. If None, use internal codebooks.

        **kwargs : additional keyword arguments
            See `this page`_ for details.

        Returns
        -------
        xq : Tensor [shape=(..., M+1)]
            Quantized vectors.

        indices : Tensor [shape=(..., Q)]
            Codebook indices.

        losses : Tensor [shape=(Q,)]
            Commitment losses.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([-0.5206,  1.0048, -0.3370,  1.3364, -0.2933])
        >>> msvq = diffsptk.MultiStageVectorQuantization(4, 3, 2).eval()
        >>> xq, indices, _ = msvq(x)
        >>> xq
        tensor([-0.4561,  0.9835, -0.3787, -0.1488, -0.8025])
        >>> indices
        tensor([0, 2])

        """
        if codebooks is not None:
            cb = self.codebooks
            for i, layer in enumerate(self.vq.layers):
                layer._codebook.embed[:] = codebooks.view_as(cb)[i]

        d = x.dim()
        if d == 1:
            x = x.unsqueeze(0)

        xq, indices, losses = self.vq(x.float(), **kwargs)

        if d == 1:
            xq = xq.squeeze(0)
            indices = indices.squeeze(0)
        losses = losses.squeeze()

        return xq, indices, losses
