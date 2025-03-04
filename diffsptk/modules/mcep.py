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

from ..misc.utils import check_size
from ..misc.utils import get_layer
from ..misc.utils import get_values
from ..misc.utils import hankel
from ..misc.utils import symmetric_toeplitz
from ..misc.utils import to
from .base import BaseFunctionalModule
from .freqt import FrequencyTransform


class MelCepstralAnalysis(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mgcep.html>`_
    for details. Note that the current implementation does not use the efficient
    Toeplitz-plus-Hankel system solver.

    Parameters
    ----------
    fft_length : int >= 2M
        The number of FFT bins, :math:`L`.

    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    n_iter : int >= 0
        The number of iterations.

    """

    def __init__(self, *, fft_length, cep_order, alpha=0, n_iter=0):
        super().__init__()

        self.in_dim = fft_length // 2 + 1

        self.values, layers, tensors = self._precompute(*get_values(locals()))
        self.layers = nn.ModuleList(layers)
        self.register_buffer("alpha_vector", tensors[0])

    def forward(self, x):
        """Perform mel-cepstral analysis.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            The power spectrum.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The mel-cepstrum.

        Examples
        --------
        >>> x = diffsptk.ramp(19)
        >>> stft = diffsptk.STFT(frame_length=10, frame_period=10, fft_length=16)
        >>> mcep = diffsptk.MelCepstralAnalysis(3, 16, 0.1, n_iter=1)
        >>> mc = mcep(stft(x))
        >>> mc
        tensor([[-0.8851,  0.7917, -0.1737,  0.0175],
                [-0.3522,  4.4222, -1.0882, -0.0511]])

        """
        check_size(x.size(-1), self.in_dim, "dimension of spectrum")
        return self._forward(x, *self.values, *self.layers, **self._buffers)

    @staticmethod
    def _func(x, *args, **kwargs):
        values, layers, tensors = MelCepstralAnalysis._precompute(
            2 * x.size(-1) - 2, *args, **kwargs, dtype=x.dtype, device=x.device
        )
        return MelCepstralAnalysis._forward(x, *values, *layers, *tensors)

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(fft_length, cep_order, alpha, n_iter):
        if fft_length <= 1:
            raise ValueError("fft_length must be greater than 1.")
        if cep_order < 0:
            raise ValueError("cep_order must be non-negative.")
        if fft_length < 2 * cep_order:
            raise ValueError("cep_order must be less than or equal to fft_length // 2.")
        if 1 <= abs(alpha):
            raise ValueError("alpha must be in (-1, 1).")
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative.")

    @staticmethod
    def _precompute(fft_length, cep_order, alpha, n_iter, device=None, dtype=None):
        MelCepstralAnalysis._check(fft_length, cep_order, alpha, n_iter)
        module = inspect.stack()[1].function == "__init__"

        freqt = get_layer(
            module,
            FrequencyTransform,
            dict(
                in_order=fft_length // 2,
                out_order=cep_order,
                alpha=alpha,
            ),
        )
        ifreqt = get_layer(
            module,
            FrequencyTransform,
            dict(
                in_order=cep_order,
                out_order=fft_length // 2,
                alpha=-alpha,
            ),
        )
        rfreqt = get_layer(
            module,
            CoefficientsFrequencyTransform,
            dict(
                in_order=fft_length // 2,
                out_order=2 * cep_order,
                alpha=alpha,
            ),
        )

        alpha_vector = (-alpha) ** torch.arange(
            cep_order + 1, device=device, dtype=dtype
        )

        return (
            (
                fft_length,
                n_iter,
            ),
            (
                freqt,
                ifreqt,
                rfreqt,
            ),
            (alpha_vector,),
        )

    @staticmethod
    def _forward(x, fft_length, n_iter, freqt, ifreqt, rfreqt, alpha_vector):
        M = len(alpha_vector) - 1
        H = fft_length // 2

        log_x = torch.log(x)
        c = torch.fft.irfft(log_x)
        c[..., 0] *= 0.5
        c[..., H] *= 0.5
        mc = freqt(c[..., : H + 1])

        for _ in range(n_iter):
            c = ifreqt(mc)
            d = torch.fft.rfft(c, n=fft_length).real
            d = torch.exp(log_x - d - d)

            rd = torch.fft.irfft(d)
            rt = rfreqt(rd[..., : H + 1])
            r = rt[..., : M + 1]
            ra = r - alpha_vector

            R = symmetric_toeplitz(r)
            Q = hankel(rt)
            gradient = torch.linalg.solve(R + Q, ra)
            mc = mc + gradient

        return mc


class CoefficientsFrequencyTransform(BaseFunctionalModule):
    def __init__(self, in_order, out_order, alpha=0):
        super().__init__()

        self.in_dim = in_order + 1

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("A", tensors[0])

    def forward(self, c):
        check_size(c.size(-1), self.in_dim, "dimension of cepstrum")
        return self._forward(c, **self._buffers)

    @staticmethod
    def _func(c, *args, **kwargs):
        _, _, tensors = CoefficientsFrequencyTransform._precompute(
            c.size(-1) - 1, *args, **kwargs, device=c.device, dtype=c.dtype
        )
        return CoefficientsFrequencyTransform._forward(c, *tensors)

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(in_order, out_order, alpha):
        if in_order < 0:
            raise ValueError("in_order must be non-negative.")
        if out_order < 0:
            raise ValueError("out_order must be non-negative.")
        if 1 <= abs(alpha):
            raise ValueError("alpha must be in (-1, 1).")

    @staticmethod
    def _precompute(in_order, out_order, alpha, device=None, dtype=None):
        CoefficientsFrequencyTransform._check(in_order, out_order, alpha)
        L1 = in_order + 1
        L2 = out_order + 1

        A = torch.zeros((L2, L1), device=device, dtype=torch.double)
        A[:, 0] = (-alpha) ** torch.arange(L2, device=device, dtype=torch.double)
        for i in range(1, L2):
            i1 = i - 1
            for j in range(1, L1):
                j1 = j - 1
                A[i, j] = A[i1, j1] + alpha * (A[i, j1] - A[i1, j])

        return None, None, (to(A.T, dtype=dtype),)

    @staticmethod
    def _forward(c, A):
        return torch.matmul(c, A)
