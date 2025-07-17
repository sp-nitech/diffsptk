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
import logging

import torch
from torch import nn

from ..typing import Callable, Precomputed
from ..utils.private import TAU, filter_values, get_layer, get_logger
from .base import BaseFunctionalModule
from .istft import InverseShortTimeFourierTransform
from .stft import ShortTimeFourierTransform


class GriffinLim(BaseFunctionalModule):
    """Griffin-Lim phase reconstruction module.

    Parameters
    ----------
    frame_length : int >= 1
        The frame length in samples, :math:`L`.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    fft_length : int >= L
        The number of FFT bins, :math:`N`.

    center : bool
        If True, pad the input on both sides so that the frame is centered.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular', 'nuttall']
        The window type.

    norm : ['none', 'power', 'magnitude']
        The normalization type of the window.

    symmetric : bool
        If True, the window is symmetric, otherwise periodic.

    n_iter : int >= 0
        The number of iterations for phase reconstruction.

    alpha : float >= 0
        The momentum factor, :math:`\\alpha`.

    beta : float >= 0
        The momentum factor, :math:`\\beta`.

    gamma : float >= 0
        The smoothing factor, :math:`\\gamma`.

    init_phase : ['zeros', 'random']
        The initial phase for the reconstruction.

    verbose : bool
        If True, print the SNR at each iteration.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] R. Nenov et al., "Faster than fast: Accelerating the Griffin-Lim algorithm,"
           *Proceedings of ICASSP*, 2023.

    """

    def __init__(
        self,
        frame_length: int,
        frame_period: int,
        fft_length: int,
        *,
        center: bool = True,
        mode: str = "constant",
        window: str = "blackman",
        norm: str = "power",
        symmetric: bool = True,
        n_iter: int = 100,
        alpha: float = 0.99,
        beta: float = 0.99,
        gamma: float = 1.1,
        init_phase: str = "random",
        verbose: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.values, layers, _ = self._precompute(**filter_values(locals()))
        self.layers = nn.ModuleList(layers)

    def forward(self, y: torch.Tensor, out_length: int | None = None) -> torch.Tensor:
        """Reconstruct a waveform from the spectrum using the Griffin-Lim algorithm.

        Parameters
        ----------
        y : Tensor [shape=(..., T/P, N/2+1)]
            The power spectrum.

        out_length : int > 0 or None
            The length of the output waveform.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The reconstructed waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 3)
        >>> x
        tensor([1., 2., 3.])
        >>> stft_params = {"frame_length": 3, "frame_period": 1, "fft_length": 8}
        >>> stft = diffsptk.STFT(**stft_params, out_format="power")
        >>> griffin = diffsptk.GriffinLim(**stft_params, n_iter=10, init_phase="zeros")
        >>> y = griffin(stft(x), out_length=3)
        >>> y
        tensor([ 1.0000,  2.0000, -3.0000])

        """
        return self._forward(y, out_length, *self.values, *self.layers)

    @staticmethod
    def _func(y: torch.Tensor, out_length: int | None, *args, **kwargs) -> torch.Tensor:
        values, layers, _ = GriffinLim._precompute(
            *args, **kwargs, device=y.device, dtype=y.dtype
        )
        return GriffinLim._forward(y, out_length, *values, *layers)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(
        n_iter: int,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> None:
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative.")
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")

    @staticmethod
    def _precompute(
        frame_length: int,
        frame_period: int,
        fft_length: int,
        center: bool,
        mode: str,
        window: str,
        norm: str,
        symmetric: bool,
        n_iter: int,
        alpha: float,
        beta: float,
        gamma: float,
        init_phase: str,
        verbose: bool,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        GriffinLim._check(n_iter, alpha, beta, gamma)
        module = inspect.stack()[1].function != "_func"

        if init_phase == "zeros":
            phase_generator = lambda x: torch.zeros_like(x)
        elif init_phase == "random":
            phase_generator = lambda x: TAU * torch.rand_like(x)
        else:
            raise ValueError(f"init_phase: {init_phase} is not supported.")

        if verbose:
            logger = get_logger("griffin")
        else:
            logger = None

        stft = get_layer(
            module,
            ShortTimeFourierTransform,
            dict(
                frame_length=frame_length,
                frame_period=frame_period,
                fft_length=fft_length,
                center=center,
                zmean=False,
                mode=mode,
                window=window,
                norm=norm,
                symmetric=symmetric,
                eps=0,
                relative_floor=None,
                out_format="complex",
                device=device,
                dtype=dtype,
            ),
        )
        istft = get_layer(
            module,
            InverseShortTimeFourierTransform,
            dict(
                frame_length=frame_length,
                frame_period=frame_period,
                fft_length=fft_length,
                center=center,
                window=window,
                norm=norm,
                symmetric=symmetric,
                device=device,
                dtype=dtype,
            ),
        )
        return (
            (n_iter, alpha, beta, gamma, phase_generator, logger),
            (stft, istft),
            None,
        )

    @staticmethod
    def _forward(
        y: torch.Tensor,
        out_length: int | None,
        n_iter: int,
        alpha: float,
        beta: float,
        gamma: float,
        phase_generator: Callable,
        logger: logging.Logger | None,
        stft: Callable,
        istft: Callable,
    ) -> torch.Tensor:
        if logger is not None:
            logger.info(f"alpha: {alpha}, beta: {beta}, gamma: {gamma}")

        eps = 1e-16
        s = torch.sqrt(y + eps)
        angle = torch.exp(1j * phase_generator(s))

        t_prev = d_prev = 0  # This suppresses F821 and F841.
        for n in range(n_iter):
            t = stft(istft(s * angle, out_length=out_length))
            t = t[..., : s.shape[-2], :]

            if 0 == n:
                c = d = t
            else:
                t = (1 - gamma) * d_prev + gamma * t
                diff = t - t_prev
                c = t + alpha * diff
                d = t + beta * diff

            angle = c / (c.abs() + eps)
            t_prev = t
            d_prev = d

            if logger is not None:
                snr = -10 * torch.log10(
                    torch.linalg.norm(c.abs() - s) / torch.linalg.norm(s)
                )
                logger.info(f"  iter {n + 1:5d}: SNR = {snr:g}")

        return istft(s * angle, out_length=out_length)
