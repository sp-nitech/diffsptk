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
import torch.nn as nn

from ..misc.utils import Lambda
from ..misc.utils import check_size
from ..misc.utils import get_gamma
from .linear_intpl import LinearInterpolation
from .mgc2mgc import MelGeneralizedCepstrumToMelGeneralizedCepstrum


def mirror(x, half=False):
    x0, x1 = torch.split(x, [1, x.size(-1) - 1], dim=-1)
    if half:
        x1 = x1 * 0.5
    x = torch.cat((x1.flip(-1), x0, x1), dim=-1)
    return x


class PseudoMGLSADigitalFilter(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mglsadf.html>`_
    for details.

    Parameters
    ----------
    filter_order : int >= 0 [scalar]
        Order of filter coefficients, :math:`M`.

    alpha : float [-1 < alpha < 1]
        Frequency warping factor, :math:`\\alpha`.

    gamma : float [-1 <= gamma <= 1]
        Gamma, :math:`\\gamma`.

    c : int >= 1 [scalar]
        Number of stages.

    frame_period : int >= 1 [scalar]
        Frame period, :math:`P`.

    ignore_gain : bool [scalar]
        If True, perform filtering without gain.

    phase : ['minimum', 'maximum', 'zero']
        Filter type.

    cascade : bool [scalar]
        If True, use multi-stage FIR filter.

    taylor_order : int >= 0 [scalar]
        Order of Taylor series expansion (valid only if **cascade** is True).

    ir_length : int >= 1 [scalar]
        Length of impulse response (valid only if **cascade** is False).

    n_fft : int >= 1 [scalar]
        Number of FFT bins for conversion (valid only if **cascade** is False).

    cep_order : int >= 0 [scalar]
        Order of linear cepstrum (used to convert input to cepstrum).

    References
    ----------
    .. [1] T. Yoshimura et al., "Embedding a differentiable mel-cepstral synthesis
           filter to a neural speech synthesis system," *arXiv:2211.11222*, 2022.

    """

    def __init__(
        self,
        filter_order,
        alpha=0,
        gamma=0,
        c=None,
        frame_period=1,
        ignore_gain=False,
        phase="minimum",
        cascade=True,
        **kwargs,
    ):
        super(PseudoMGLSADigitalFilter, self).__init__()

        self.filter_order = filter_order
        self.frame_period = frame_period

        gamma = get_gamma(gamma, c)

        if cascade:
            self.mglsadf = MultiStageFIRFilter(
                filter_order,
                alpha=alpha,
                gamma=gamma,
                frame_period=frame_period,
                ignore_gain=ignore_gain,
                phase=phase,
                **kwargs,
            )
        else:
            self.mglsadf = SingleStageFIRFilter(
                filter_order,
                alpha=alpha,
                gamma=gamma,
                frame_period=frame_period,
                ignore_gain=ignore_gain,
                phase=phase,
                **kwargs,
            )

    def forward(self, x, mc):
        """Apply an MGLSA digital filter.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Excitation signal.

        mc : Tensor [shape=(..., T/P, M+1)]
            Mel-generalized cepstrum, not MLSA digital filter coefficients.

        Returns
        -------
        y : Tensor [shape=(..., T)]
            Output signal.

        Examples
        --------
        >>> M = 4
        >>> x = diffsptk.step(3)
        >>> mc = diffsptk.nrand(2, M)
        >>> mc
        tensor([[-0.9134, -0.5774, -0.4567,  0.7423, -0.5782],
                [ 0.6904,  0.5175,  0.8765,  0.1677,  2.4624]])
        >>> mglsadf = diffsptk.MLSA(M, frame_period=2)
        >>> y = mglsadf(x.view(1, -1), mc.view(1, 2, M + 1))
        >>> y
        tensor([[0.4011, 0.8760, 3.5677, 4.8725]])

        """
        check_size(mc.size(-1), self.filter_order + 1, "dimension of mel-cepstrum")
        check_size(x.size(-1), mc.size(-2) * self.frame_period, "sequence length")

        y = self.mglsadf(x, mc)
        return y


class MultiStageFIRFilter(nn.Module):
    def __init__(
        self,
        filter_order,
        alpha=0,
        gamma=0,
        frame_period=1,
        ignore_gain=False,
        phase="minimum",
        taylor_order=20,
        cep_order=199,
    ):
        super(MultiStageFIRFilter, self).__init__()

        self.ignore_gain = ignore_gain
        self.phase = phase
        self.taylor_order = taylor_order

        assert 0 <= self.taylor_order

        if self.phase == "minimum":
            self.pad = nn.ConstantPad1d((cep_order, 0), 0)
        elif self.phase == "maximum":
            self.pad = nn.ConstantPad1d((0, cep_order), 0)
        elif self.phase == "zero":
            self.pad = nn.ConstantPad1d((cep_order, cep_order), 0)
        else:
            raise ValueError(f"phase {phase} is not supported")

        self.mgc2c = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
            filter_order,
            cep_order,
            in_alpha=alpha,
            in_gamma=gamma,
        )
        self.linear_intpl = LinearInterpolation(frame_period)

    def forward(self, x, mc):
        c = self.mgc2c(mc)
        if self.ignore_gain:
            c[..., 0] = 0

        if self.phase == "minimum":
            c = c.flip(-1)
        elif self.phase == "maximum":
            pass
        elif self.phase == "zero":
            c = mirror(c, half=True)
        else:
            raise RuntimeError

        c = self.linear_intpl(c)

        y = x.clone()
        for a in range(1, self.taylor_order + 1):
            x = self.pad(x)
            x = x.unfold(-1, c.size(-1), 1)
            x = (x * c).sum(-1) / a
            y += x
        return y


class SingleStageFIRFilter(nn.Module):
    def __init__(
        self,
        filter_order,
        alpha=0,
        gamma=0,
        frame_period=1,
        ignore_gain=False,
        phase="minimum",
        ir_length=2000,
        n_fft=4096,
        cep_order=199,
    ):
        super(SingleStageFIRFilter, self).__init__()

        self.ignore_gain = ignore_gain
        self.phase = phase

        taps = ir_length - 1
        if self.phase == "minimum":
            self.pad = nn.ConstantPad1d((taps, 0), 0)
        elif self.phase == "maximum":
            self.pad = nn.ConstantPad1d((0, taps), 0)
        elif self.phase == "zero":
            self.pad = nn.ConstantPad1d((taps, taps), 0)
        else:
            raise ValueError(f"phase {phase} is not supported")

        if self.phase in ["minimum", "maximum"]:
            self.mgc2ir = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
                filter_order,
                ir_length - 1,
                in_alpha=alpha,
                in_gamma=gamma,
                out_gamma=1,
                out_mul=True,
                n_fft=n_fft,
            )
        else:
            self.mgc2c = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
                filter_order,
                cep_order,
                in_alpha=alpha,
                in_gamma=gamma,
            )
            self.c2ir = nn.Sequential(
                Lambda(lambda x: torch.fft.hfft(x, n=n_fft)),
                Lambda(lambda x: torch.fft.ifft(torch.exp(x)).real[..., :ir_length]),
            )
        self.linear_intpl = LinearInterpolation(frame_period)

    def forward(self, x, mc):
        if self.phase == "zero":
            c = self.mgc2c(mc)
            c[..., 1:] *= 0.5
            if self.ignore_gain:
                c[..., 0] = 0
            h = self.c2ir(c)
        else:
            h = self.mgc2ir(mc)

        if self.phase == "minimum":
            h = h.flip(-1)
        elif self.phase == "maximum":
            pass
        elif self.phase == "zero":
            h = mirror(h)
        else:
            raise RuntimeError

        # FIXME: linear interpolation of impulse response seems inappropriate.
        h = self.linear_intpl(h)

        if self.ignore_gain:
            if self.phase == "minimum":
                h = h / h[..., -1:]
            elif self.phase == "maximum":
                h = h / h[..., :1]

        x = self.pad(x)
        x = x.unfold(-1, h.size(-1), 1)
        y = (x * h).sum(-1)
        return y
