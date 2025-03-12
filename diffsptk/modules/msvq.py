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

from .base import BaseNonFunctionalModule


class MultiStageVectorQuantization(BaseNonFunctionalModule):
    """See `this page <https://github.com/lucidrains/vector-quantize-pytorch>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        The order of the input vector, :math:`M`.

    codebook_size : int >= 1
        The codebook size, :math:`K`.

    n_stage : int >= 1
        The number of stages (quantizers), :math:`Q`.

    **kwargs : additional keyword arguments
        See `this page`_ for details.

    References
    ----------
    .. [1] A. v. d. Oord et al., "Neural discrete representation learning," *Advances in
           Neural Information Processing Systems*, pp. 6309-6318, 2017.

    """

    def __init__(self, order, codebook_size, n_stage, **kwargs):
        super().__init__()

        if order < 0:
            raise ValueError("order must be non-negative.")
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive.")
        if n_stage <= 0:
            raise ValueError("n_stage must be positive.")

        from vector_quantize_pytorch import ResidualVQ

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
            The input vectors.

        codebooks : Tensor [shape=(Q, K, M+1)]
            The external codebook. If None, use the internal codebook.

        **kwargs : additional keyword arguments
            See `this page`_ for details.

        Returns
        -------
        xq : Tensor [shape=(..., M+1)]
            The quantized vectors.

        indices : Tensor [shape=(..., Q)]
            The codebook indices.

        losses : Tensor [shape=(Q,)]
            The commitment losses.

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
