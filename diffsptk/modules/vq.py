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

import warnings

from torch import nn

warnings.simplefilter("ignore", UserWarning)
from vector_quantize_pytorch import VectorQuantize  # noqa: E402


class VectorQuantization(nn.Module):
    """See `this page <https://github.com/lucidrains/vector-quantize-pytorch>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        Order of vector, :math:`M`.

    codebook_size : int >= 1
        Codebook size, :math:`K`.

    **kwargs : additional keyword arguments
        See `this page`_ for details.

    """

    def __init__(self, order, codebook_size, **kwargs):
        super().__init__()

        assert 0 <= order
        assert 1 <= codebook_size

        self.vq = VectorQuantize(
            dim=order + 1, codebook_size=codebook_size, **kwargs
        ).float()

    @property
    def codebook(self):
        return self.vq.codebook

    def forward(self, x, codebook=None, **kwargs):
        """Perform vector quantization.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            Input vectors.

        codebook : Tensor [shape=(K, M+1)]
            External codebook. If None, use internal codebook.

        **kwargs : additional keyword arguments
            See `this page`_ for details.

        Returns
        -------
        xq : Tensor [shape=(..., M+1)]
            Quantized vectors.

        indices : Tensor [shape=(...,)]
            Codebook indices.

        loss : Tensor [scalar]
            Commitment loss.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([ 0.7947,  0.1007,  1.2290, -0.5019,  1.5552])
        >>> vq = diffsptk.VectorQuantization(4, 2).eval()
        >>> xq, _, _ = vq(x)
        >>> xq
        tensor([0.3620, 0.2736, 0.7098, 0.7106, 0.6494]

        """
        if codebook is not None:
            self.codebook[:] = codebook.view_as(self.vq.codebook)

        d = x.dim()
        if d == 1:
            x = x.unsqueeze(0)

        xq, indices, loss = self.vq(x.float(), **kwargs)

        if d == 1:
            xq = xq.squeeze(0)
            indices = indices.squeeze(0)
        loss = loss.squeeze()

        return xq, indices, loss
