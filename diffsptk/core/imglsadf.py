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

import torch.nn as nn

from .mglsadf import PseudoMGLSADigitalFilter


class PseudoInverseMGLSADigitalFilter(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/imglsadf.html>`_
    for details.

    Parameters
    ----------
    filter_order : int >= 0 [scalar]
        Order of filter coefficients, :math:`M`.

    frame_period : int >= 1 [scalar]
        Frame period, :math:`P`.

    alpha : float [-1 < alpha < 1]
        Frequency warping factor, :math:`\\alpha`.

    gamma : float [-1 <= gamma <= 1]
        Gamma, :math:`\\gamma`.

    c : int >= 1 [scalar]
        Number of stages.

    ignore_gain : bool [scalar]
        If True, perform filtering without gain.

    phase : ['minimum', 'maximum', 'zero']
        Filter type.

    mode : ['multi-stage', 'single-stage', 'freq-domain']
        'multi-stage' approximates the MLSA filter by cascading FIR filters based on the
        Taylor series expansion. 'single-stage' uses a FIR filter whose coefficients are
        the impulse response converted from mel-cepstral coefficients. 'freq-domain'
        performs filtering in the frequency domain rather than time one.

    taylor_order : int >= 0 [scalar]
        Order of Taylor series expansion (valid only if **mode** is 'multi-stage').

    cep_order : int >= 0 [scalar]
        Order of linear cepstrum (valid only if **mode** is 'multi-stage').

    ir_length : int >= 1 [scalar]
        Length of impulse response (valid only if **mode** is 'single-stage').

    n_fft : int >= 1 [scalar]
        Number of FFT bins for conversion (valid only if **mode** is 'single-stage').

    **stft_kwargs : additional keyword arguments
        See ShortTermFourierTransform (valid only if **mode** is 'freq-domain').

    """

    def __init__(self, filter_order, frame_period, **kwargs):
        super(PseudoInverseMGLSADigitalFilter, self).__init__()

        # Change the default value of the order of Taylor series.
        # This is because inverse filtering requires the large value.
        if (
            kwargs.get("mode", "multi-stage") == "multi-stage"
            and "taylor_order" not in kwargs
        ):
            kwargs["taylor_order"] = 40

        self.mglsadf = PseudoMGLSADigitalFilter(filter_order, frame_period, **kwargs)

    def forward(self, y, mc):
        """Apply an inverse MGLSA digital filter.

        Parameters
        ----------
        y : Tensor [shape=(..., T)]
            Audio signal.

        mc : Tensor [shape=(..., T/P, M+1)]
            Mel-generalized cepstrum, not MLSA digital filter coefficients.

        Returns
        -------
        x : Tensor [shape=(..., T)]
            Residual signal.

        Examples
        --------
        >>> M = 4
        >>> y = diffsptk.step(3)
        >>> mc = diffsptk.nrand(2, M)
        >>> mc
        tensor([[ 0.8457,  1.5812,  0.1379,  1.6558,  1.4591],
                [-1.3714, -0.9669, -1.2025, -1.3683, -0.2352]])
        >>> imglsadf = diffsptk.IMLSA(M, frame_period=2)
        >>> x = imglsadf(y.view(1, -1), mc.view(1, 2, M + 1))
        >>> x
        tensor([[ 0.4293,  1.0592,  7.9349, 14.9794]])

        """
        x = self.mglsadf(y, -mc)
        return x
