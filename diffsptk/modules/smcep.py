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

from ..utils.private import check_size
from ..utils.private import get_layer
from ..utils.private import get_values
from ..utils.private import to
from .base import BaseFunctionalModule
from .freqt2 import SecondOrderAllPassFrequencyTransform
from .ifreqt2 import SecondOrderAllPassInverseFrequencyTransform
from .mcep import MelCepstralAnalysis


class SecondOrderAllPassMelCepstralAnalysis(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/smcep.html>`_
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

    theta : float in [0, 1]
        The emphasis frequency, :math:`\\theta`.

    n_iter : int >= 0
        The number of iterations.

    accuracy_factor : int >= 1
        The accuracy factor multiplied by the FFT length.

    References
    ----------
    .. [1] T. Wakako et al., "Speech spectral estimation based on expansion of log
           spectrum by arbitrary basis functions," *IEICE Trans*, vol. J82-D-II, no. 12,
           pp. 2203-2211, 1999 (in Japanese).

    """

    def __init__(
        self, *, fft_length, cep_order, alpha=0, theta=0, n_iter=0, accuracy_factor=4
    ):
        super().__init__()

        self.in_dim = fft_length // 2 + 1

        self.values, layers, tensors = self._precompute(*get_values(locals()))
        self.layers = nn.ModuleList(layers)
        self.register_buffer("alpha_vector", tensors[0])

    def forward(self, x):
        """Perform mel-cepstral analysis based on the second-order all-pass filter.

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
        >>> smcep = diffsptk.SecondOrderAllPassMelCepstralAnalysis(
        ...     fft_length=16, cep_order=3, alpha=0.1, n_iter=1
        ... )
        >>> mc = smcep(stft(x))
        >>> mc
        tensor([[-0.8851,  0.7917, -0.1737,  0.0175],
                [-0.3523,  4.4223, -1.0883, -0.0510]])

        """
        check_size(x.size(-1), self.in_dim, "dimension of spectrum")
        return self._forward(x, *self.values, *self.layers, **self._buffers)

    @staticmethod
    def _func(x, *args, **kwargs):
        values, layers, tensors = SecondOrderAllPassMelCepstralAnalysis._precompute(
            2 * x.size(-1) - 2, *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return SecondOrderAllPassMelCepstralAnalysis._forward(
            x, *values, *layers, *tensors
        )

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(fft_length, cep_order, alpha, theta, n_iter, accuracy_factor):
        if fft_length <= 1:
            raise ValueError("fft_length must be greater than 1.")
        if cep_order < 0:
            raise ValueError("cep_order must be non-negative.")
        if fft_length < 2 * cep_order:
            raise ValueError("cep_order must be less than or equal to fft_length // 2.")
        if 1 <= abs(alpha):
            raise ValueError("alpha must be in (-1, 1).")
        if not 0 <= theta <= 1:
            raise ValueError("theta must be in [0, 1].")
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative.")
        if accuracy_factor <= 0:
            raise ValueError("accuracy_factor must be positive.")

    @staticmethod
    def _precompute(
        fft_length,
        cep_order,
        alpha,
        theta,
        n_iter,
        accuracy_factor,
        device=None,
        dtype=None,
    ):
        SecondOrderAllPassMelCepstralAnalysis._check(
            fft_length, cep_order, alpha, theta, n_iter, accuracy_factor
        )
        module = inspect.stack()[1].function == "__init__"

        n_fft = fft_length * accuracy_factor
        freqt = get_layer(
            module,
            SecondOrderAllPassFrequencyTransform,
            dict(
                in_order=fft_length // 2,
                out_order=cep_order,
                alpha=alpha,
                theta=theta,
                n_fft=n_fft,
            ),
        )
        ifreqt = get_layer(
            module,
            SecondOrderAllPassInverseFrequencyTransform,
            dict(
                in_order=cep_order,
                out_order=fft_length // 2,
                alpha=alpha,
                theta=theta,
                n_fft=n_fft,
            ),
        )
        rfreqt = get_layer(
            module,
            CoefficientsFrequencyTransform,
            dict(
                in_order=fft_length // 2,
                out_order=2 * cep_order,
                alpha=alpha,
                theta=theta,
                n_fft=n_fft,
            ),
        )

        seed = to(torch.ones(1, device=device), dtype=dtype)
        alpha_vector = CoefficientsFrequencyTransform._func(
            seed,
            out_order=cep_order,
            alpha=alpha,
            theta=theta,
            n_fft=n_fft,
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
    def _forward(*args, **kwargs):
        return MelCepstralAnalysis._forward(*args, **kwargs)


class CoefficientsFrequencyTransform(BaseFunctionalModule):
    def __init__(self, in_order, out_order, alpha=0, theta=0, n_fft=512):
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
    def _check(in_order, out_order, alpha, theta, n_fft):
        if in_order < 0:
            raise ValueError("in_order must be non-negative.")
        if out_order < 0:
            raise ValueError("out_order must be non-negative.")
        if 1 <= abs(alpha):
            raise ValueError("alpha must be in (-1, 1).")
        if not 0 <= theta <= 1:
            raise ValueError("theta must be in [0, 1].")
        if n_fft <= 1:
            raise ValueError("n_fft must be greater than 1.")

    @staticmethod
    def _precompute(in_order, out_order, alpha, theta, n_fft, device=None, dtype=None):
        CoefficientsFrequencyTransform._check(in_order, out_order, alpha, theta, n_fft)
        theta *= torch.pi
        k = torch.arange(n_fft, device=device, dtype=torch.double)
        omega = k * (2 * torch.pi / n_fft)
        ww = SecondOrderAllPassFrequencyTransform.warp(omega, alpha, theta)

        m2 = k[: out_order + 1]
        wwm2 = ww.reshape(-1, 1) * m2.reshape(1, -1)
        real = torch.cos(wwm2)
        imag = -torch.sin(wwm2)

        A = torch.fft.ifft(torch.complex(real, imag), dim=0).real
        L = in_order + 1
        if 2 <= L:
            A[1:L] += A[-(L - 1) :].flip(0)
        A = A[:L]
        return None, None, (to(A, dtype=dtype),)

    @staticmethod
    def _forward(c, A):
        return torch.matmul(c, A)
