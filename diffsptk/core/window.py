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


class Window(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/window.html>`_
    for details.

    Parameters
    ----------
    in_length : int >= 1 [scalar]
        Input length or window length, :math:`L_1`.

    out_length : int >= L1 [scalar]
        Output length, :math:`L_2`. If :math:`L_2 > L_1`, output is zero-padded.

    norm : ['none', 'power', 'magnitude']
        Normalization type of window.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
        'rectangular']
        Window type.

    """

    def __init__(self, in_length, out_length=None, norm="power", window="blackman"):
        super(Window, self).__init__()

        assert 1 <= in_length

        # Make window.
        if window == 0 or window == "blackman":
            w = np.blackman(in_length)
        elif window == 1 or window == "hamming":
            w = np.hamming(in_length)
        elif window == 2 or window == "hanning":
            w = np.hanning(in_length)
        elif window == 3 or window == "bartlett":
            w = np.bartlett(in_length)
        elif window == 4 or window == "trapezoidal":
            slope = np.linspace(0, 4, in_length)
            w = np.minimum(np.clip(slope, 0, 1), np.flip(slope))
        elif window == 5 or window == "rectangular":
            w = np.ones(in_length)
        else:
            raise ValueError(f"window {window} is not supported")

        # Normalize window.
        if norm == 0 or norm == "none":
            pass
        elif norm == 1 or norm == "power":
            w /= np.sqrt(np.sum(w**2))
        elif norm == 2 or norm == "magnitude":
            w /= np.sum(w)
        else:
            raise ValueError(f"norm {norm} is not supported")

        self.register_buffer("window", torch.from_numpy(w.astype(np.float32)))

        # Make padding module.
        if out_length is None or in_length == out_length:
            self.pad = lambda x: x
        else:
            assert in_length <= out_length
            self.pad = nn.ConstantPad1d((0, out_length - in_length), 0)

    def forward(self, x):
        """Apply a window function to given waveform.

        Parameters
        ----------
        x : Tensor [shape=(..., L1)]
            Framed waveform.

        Returns
        -------
        y : Tensor [shape=(..., L2)]
            Windowed waveform.

        Examples
        --------
        >>> x = torch.ones(5)
        >>> window = diffsptk.Window(5, out_length=7, window="hamming", norm="none")
        >>> y = window(x)
        >>> y
        tensor([0.0800, 0.5400, 1.0000, 0.5400, 0.0800, 0.0000, 0.0000])

        """
        y = self.pad(x * self.window)
        return y
