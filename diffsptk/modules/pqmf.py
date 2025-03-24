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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..utils.private import next_power_of_two, numpy_to_torch
from .base import BaseNonFunctionalModule


def make_filter_banks(
    n_band: int,
    filter_order: int,
    mode: str = "analysis",
    alpha: float = 100,
    n_iter: int = 100,
    step_size: float = 1e-2,
    decay: float = 0.5,
    eps: float = 1e-6,
) -> tuple[np.ndarray, bool]:
    """Make filter-bank coefficients.

    Parameters
    ----------
    n_band : int >= 1
        The number of subbands, :math:`K`.

    filter_order : int >= 2
        The order of the filters, :math:`M`.

    mode : ['analysis' or 'synthesis']
        Analysis or synthesis.

    alpha : float > 0
        The stopband attenuation in dB.

    n_iter : int >= 1
        The number of iterations to find optimal filter-bank coefficients.

    step_size : float > 0
        The step size of optimization.

    decay : float > 0
        The decay factor of the step size.

    eps : float >= 0
        The convergence criterion.

    Returns
    -------
    filters : ndarray [shape=(K, M + 1)]
        The filter-bank coefficients.

    is_converged : bool
        Whether the optimization is converged.

    """
    if n_band <= 0:
        raise ValueError("n_band must be positive.")
    if filter_order <= 1:
        raise ValueError("filter_order must be greater than or equal to 2.")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if decay <= 0:
        raise ValueError("decay must be positive.")
    if eps < 0:
        raise ValueError("eps must be non-negative.")

    def alpha_to_beta(alpha):
        if alpha <= 21:
            return 0
        elif alpha <= 50:
            a = alpha - 21
            return 0.5842 * np.power(a, 0.4) + 0.07886 * a
        else:
            a = alpha - 8.7
            return 0.1102 * a

    w = np.kaiser(filter_order + 1, alpha_to_beta(alpha))
    x = np.arange(filter_order + 1) - 0.5 * filter_order

    fft_length = next_power_of_two(filter_order + 1)
    index = fft_length // (4 * n_band)

    omega = np.pi / (2 * n_band)
    best_abs_error = np.inf

    is_converged = False
    for _ in range(n_iter):
        with np.errstate(invalid="ignore"):
            h = np.sin(omega * x) / (np.pi * x)
        if filter_order % 2 == 0:
            h[filter_order // 2] = omega / np.pi

        prototype_filter = h * w
        H = np.fft.rfft(prototype_filter, n=fft_length)

        error = np.square(np.abs(H[index])) - 0.5
        abs_error = np.abs(error)
        if abs_error < eps:
            is_converged = True
            break

        if abs_error < best_abs_error:
            best_abs_error = abs_error
            omega -= np.sign(error) * step_size
        else:
            step_size *= decay
            omega -= np.sign(error) * step_size

    if mode == "analysis":
        sign = 1
    elif mode == "synthesis":
        sign = -1
    else:
        raise ValueError("analysis or synthesis is expected.")

    filters = []
    for k in range(n_band):
        a = ((2 * k + 1) * np.pi / (2 * n_band)) * x
        b = (-1) ** k * (np.pi / 4) * sign
        c = 2 * prototype_filter
        filters.append(c * np.cos(a + b))
    filters = np.asarray(filters)

    return filters, is_converged


class PseudoQuadratureMirrorFilterBankAnalysis(BaseNonFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/pqmf.html>`_
    for details.

    Parameters
    ----------
    n_band : int >= 1
        The number of subbands, :math:`K`.

    filter_order : int >= 2
        The order of the filters, :math:`M`.

    alpha : float > 0
        The stopband attenuation in dB.

    learnable : bool
        Whether to make the filter-bank coefficients learnable.

    **kwargs : additional keyword arguments
        The parameters to find optimal filter-bank coefficients.

    References
    ----------
    .. [1] T. Q. Nguyen, "Near-perfect-reconstruction pseudo-QMF banks," *IEEE
           Transactions on Signal Processing*, vol. 42, no. 1, pp. 65-76, 1994.

    .. [2] F. Cruz-Roldan et al., "An efficient and simple method for designing
           prototype filters for cosine-modulated filter banks," *IEEE Signal
           Processing Letters*, vol. 9, no. 1, pp. 29-31, 2002.

    """

    def __init__(
        self,
        n_band: int,
        filter_order: int,
        alpha: float = 100,
        learnable: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # Make filterbanks.
        filters, is_converged = make_filter_banks(
            n_band, filter_order, mode="analysis", alpha=alpha, **kwargs
        )
        if not is_converged:
            warnings.warn("Failed to find PQMF coefficients.")
        filters = np.expand_dims(filters, 1)
        filters = np.flip(filters, 2).copy()
        filters = numpy_to_torch(filters)
        if learnable:
            self.filters = nn.Parameter(filters)
        else:
            self.register_buffer("filters", filters)

        # Make padding module.
        if filter_order % 2 == 0:
            delay_left = filter_order // 2
            delay_right = filter_order // 2
        else:
            delay_left = (filter_order + 1) // 2
            delay_right = (filter_order - 1) // 2
        self.pad = nn.Sequential(
            nn.ConstantPad1d((delay_left, 0), 0),
            nn.ReplicationPad1d((0, delay_right)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decompose waveform into subband waveforms.

        Parameters
        ----------
        x : Tensor [shape=(B, 1, T) or (B, T) or (T,)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(B, K, T)]
            The subband waveforms.

        Examples
        --------
        >>> x = diffsptk.ramp(0, 1, 0.25)
        >>> pqmf = diffsptk.PQMF(2, 10)
        >>> y = pmqf(x)
        >>> y
        tensor([[[ 0.1605,  0.4266,  0.6927,  0.9199,  1.0302],
                 [-0.0775, -0.0493, -0.0211, -0.0318,  0.0743]]])

        """
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError("Input must be 1D tensor.")

        y = F.conv1d(self.pad(x), self.filters)
        return y
