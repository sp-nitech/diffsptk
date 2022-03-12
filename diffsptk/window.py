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


class Window(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/window.html>`_
    for details.

    Parameters
    ----------
    in_length : int >= 1 [scalar]
        Input length or window length, :math:`L_1`.

    out_length : int >= L1 [scalar]
        Output length, :math:`L_2`. If :math:`L_2 > L_1`, output is zero-padded.

    norm : ['none', 'pow', 'mag']
        Normalization type of window.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
        'rectangular']
        Window type.

    """

    def __init__(self, in_length, out_length=None, norm="pow", window="blackman"):
        super(Window, self).__init__()

        self.in_length = in_length
        self.out_length = in_length if out_length is None else out_length

        if window == 0 or window == "blackman":
            w = np.blackman(self.in_length)
        elif window == 1 or window == "hamming":
            w = np.hamming(self.in_length)
        elif window == 2 or window == "hanning":
            w = np.hanning(self.in_length)
        elif window == 3 or window == "bartlett":
            w = np.bartlett(self.in_length)
        elif window == 4 or window == "trapezoidal":
            slope = np.linspace(0, 4, self.in_length)
            w = np.minimum(np.clip(slope, 0, 1), np.flip(slope))
        elif window == 5 or window == "rectangular":
            w = np.ones(self.in_length)
        else:
            raise ValueError(f"window {window} is not supported")

        if norm == 0 or norm == "none":
            pass
        elif norm == 1 or norm == "pow":
            w /= np.sqrt(np.sum(w**2))
        elif norm == 2 or norm == "mag":
            w /= np.sum(w)
        else:
            raise ValueError(f"norm {norm} is not supported")

        self.register_buffer("window", torch.from_numpy(w.astype(np.float32)))

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

        """
        w = self.window if x.dtype == self.window.dtype else self.window.double()
        y = F.pad(x * w, (0, self.out_length - self.in_length))
        return y
