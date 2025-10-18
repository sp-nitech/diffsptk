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


class VectorQuantization(BaseNonFunctionalModule):
    """See `this page <https://github.com/lucidrains/vector-quantize-pytorch>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        The order of the input vector, :math:`M`.

    codebook_size : int >= 1
        The codebook size, :math:`K`.

    device : torch.device or None
        The device of this module.

    **kwargs : additional keyword arguments
        See `this page`_ for details.

    References
    ----------
    .. [1] A. v. d. Oord et al., "Neural discrete representation learning," *Advances in
           Neural Information Processing Systems*, pp. 6309-6318, 2017.

    """

    def __init__(
        self,
        order: int,
        codebook_size: int,
        device: torch.device | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if order < 0:
            raise ValueError("order must be non-negative.")
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive.")

        from vector_quantize_pytorch import VectorQuantize

        self.vq = VectorQuantize(
            dim=order + 1, codebook_size=codebook_size, **kwargs
        ).to(device=device, dtype=torch.float)

    @property
    def codebook(self) -> torch.Tensor:
        return self.vq.codebook

    def forward(
        self, x: torch.Tensor, codebook: torch.Tensor | None = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform vector quantization.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            The input vectors.

        codebook : Tensor [shape=(K, M+1)]
            The external codebook. If None, use the internal codebook.

        **kwargs : additional keyword arguments
            See `this page`_ for details.

        Returns
        -------
        xq : Tensor [shape=(..., M+1)]
            The quantized vectors.

        indices : Tensor [shape=(...,)]
            The codebook indices.

        loss : Tensor [scalar]
            The commitment loss.

        Examples
        --------
        >>> import diffsptk
        >>> vq = diffsptk.VectorQuantization(4, 2).eval()
        >>> x = diffsptk.nrand(4)
        >>> x.shape
        torch.Size([5])
        >>> xq, _, _ = vq(x)
        >>> xq.shape
        torch.Size([5])

        """
        if codebook is not None:
            self.codebook[:] = codebook.view_as(self.vq.codebook)

        d = x.dim()
        if d == 1:
            x = x.unsqueeze(0)

        xq, indices, loss = self.vq(x.float(), **kwargs)
        xq = xq.to(dtype=x.dtype)

        if d == 1:
            xq = xq.squeeze(0)
            indices = indices.squeeze(0)
        loss = loss.squeeze()

        return xq, indices, loss
