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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..misc.utils import default_dtype
from ..misc.utils import next_power_of_two


def make_filter_banks(
    n_band,
    filter_order,
    mode="analysis",
    alpha=100,
    n_iter=100,
    step_size=1e-2,
    decay=0.5,
    eps=1e-6,
):
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
        raise ValueError("analysis or synthesis is expected")

    filters = []
    for k in range(n_band):
        a = ((2 * k + 1) * np.pi / (2 * n_band)) * x
        b = (-1) ** k * (np.pi / 4) * sign
        c = 2 * prototype_filter
        filters.append(c * np.cos(a + b))

    return np.asarray(filters, dtype=default_dtype())


class PseudoQuadratureMirrorFilterBanks(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/pqmf.html>`_
    for details.

    Parameters
    ----------
    n_band : int >= 1 [scalar]
        Number of subbands, :math:`K`.

    filter_order : int >= 2 [scalar]
        Order of filter, :math:`M`.

    alpha : float > 0 [scalar]
        Stopband attenuation in dB.

    """

    def __init__(self, n_band, filter_order, alpha=100):
        super(PseudoQuadratureMirrorFilterBanks, self).__init__()

        assert 1 <= n_band
        assert 2 <= filter_order
        assert 0 < alpha

        # Make filterbanks.
        filters = make_filter_banks(n_band, filter_order, "analysis", alpha=alpha)
        filters = np.expand_dims(filters, 1)
        filters = np.flip(filters, 2).copy()
        self.register_buffer("filters", torch.from_numpy(filters))

        # Make padding module.
        if filter_order % 2 == 0:
            delay_left = filter_order // 2
            delay_right = filter_order // 2
        else:
            delay_left = (filter_order + 1) // 2
            delay_right = (filter_order - 1) // 2

        self.pad = nn.Sequential(
            nn.ConstantPad1d((delay_left, 0), 0), nn.ReplicationPad1d((0, delay_right))
        )

    def forward(self, x):
        """Decompose waveform into subband waveforms.

        Parameters
        ----------
        x : Tensor [shape=(B, 1, T) or (B, T) or (T,) ]
            Original waveform.

        Returns
        -------
        y : Tensor [shape=(B, K, T)]
            Subband waveforms.

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
        assert x.dim() == 3, "Input must be 3D tensor"

        y = F.conv1d(self.pad(x), self.filters)
        return y
