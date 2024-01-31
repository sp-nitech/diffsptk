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
from .b2mc import MLSADigitalFilterCoefficientsToMelCepstrum
from .gnorm import GeneralizedCepstrumGainNormalization
from .istft import InverseShortTermFourierTransform
from .linear_intpl import LinearInterpolation
from .mc2b import MelCepstrumToMLSADigitalFilterCoefficients
from .mgc2mgc import MelGeneralizedCepstrumToMelGeneralizedCepstrum
from .mgc2sp import MelGeneralizedCepstrumToSpectrum
from .stft import ShortTermFourierTransform


def mirror(x, half=False):
    """Mirror the input tensor.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        Input tensor.

    half : bool [scalar]
        If True, multiply all elements except the first one by 0.5.

    Returns
    -------
    y : Tensor [shape=(..., 2L-1)]
        Output tensor.

    """
    x0, x1 = torch.split(x, [1, x.size(-1) - 1], dim=-1)
    if half:
        x1 = x1 * 0.5
    y = torch.cat((x1.flip(-1), x0, x1), dim=-1)
    return y


class PseudoMGLSADigitalFilter(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mglsadf.html>`_
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
        the impulse response converted from input mel-cepstral coefficients using FFT.
        'freq-domain' performs filtering in the frequency domain rather than time one.

    taylor_order : int >= 0 [scalar]
        Order of Taylor series expansion (valid only if **mode** is 'multi-stage').

    cep_order : int >= 0 [scalar]
        Order of linear cepstrum (valid only if **mode** is 'multi-stage').

    ir_length : int >= 1 [scalar]
        Length of impulse response (valid only if **mode** is 'single-stage').

    n_fft : int >= 1 [scalar]
        Number of FFT bins for conversion (valid only if **mode** is 'single-stage').

    **stft_kwargs : additional keyword arguments
        See :func:`~diffsptk.ShortTermFourierTransform` (valid only if **mode** is
        'freq-domain').

    References
    ----------
    .. [1] T. Yoshimura et al., "Embedding a differentiable mel-cepstral synthesis
           filter to a neural speech synthesis system," *arXiv:2211.11222*, 2022.

    """

    def __init__(
        self,
        filter_order,
        frame_period,
        *,
        alpha=0,
        gamma=0,
        c=None,
        ignore_gain=False,
        phase="minimum",
        mode="multi-stage",
        **kwargs,
    ):
        super(PseudoMGLSADigitalFilter, self).__init__()

        self.filter_order = filter_order
        self.frame_period = frame_period

        gamma = get_gamma(gamma, c)

        if mode == "multi-stage":
            self.mglsadf = MultiStageFIRFilter(
                filter_order,
                frame_period,
                alpha=alpha,
                gamma=gamma,
                ignore_gain=ignore_gain,
                phase=phase,
                **kwargs,
            )
        elif mode == "single-stage":
            self.mglsadf = SingleStageFIRFilter(
                filter_order,
                frame_period,
                alpha=alpha,
                gamma=gamma,
                ignore_gain=ignore_gain,
                phase=phase,
                **kwargs,
            )
        elif mode == "freq-domain":
            self.mglsadf = FrequencyDomainFIRFilter(
                filter_order,
                frame_period,
                alpha=alpha,
                gamma=gamma,
                ignore_gain=ignore_gain,
                phase=phase,
                **kwargs,
            )
        else:
            raise ValueError(f"mode {mode} is not supported")

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
        frame_period,
        *,
        alpha=0,
        gamma=0,
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
        frame_period,
        *,
        alpha=0,
        gamma=0,
        ignore_gain=False,
        phase="minimum",
        ir_length=2000,
        n_fft=4096,
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
                ir_length - 1,
                in_alpha=alpha,
                in_gamma=gamma,
                n_fft=n_fft,
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

        h = self.linear_intpl(h)

        if self.ignore_gain:
            if self.phase == "minimum":
                h = h / h[..., -1:]
            elif self.phase == "maximum":
                h = h / h[..., :1]
            elif self.phase == "zero":
                pass
            else:
                raise RuntimeError

        x = self.pad(x)
        x = x.unfold(-1, h.size(-1), 1)
        y = (x * h).sum(-1)
        return y


class FrequencyDomainFIRFilter(nn.Module):
    def __init__(
        self,
        filter_order,
        frame_period,
        *,
        alpha=0,
        gamma=0,
        ignore_gain=False,
        phase="minimum",
        frame_length=400,
        fft_length=512,
        n_fft=512,
        **stft_kwargs,
    ):
        super(FrequencyDomainFIRFilter, self).__init__()

        assert 2 * frame_period < frame_length

        self.ignore_gain = ignore_gain

        if self.ignore_gain:
            self.gnorm = GeneralizedCepstrumGainNormalization(filter_order, gamma=gamma)
            self.mc2b = MelCepstrumToMLSADigitalFilterCoefficients(
                filter_order, alpha=alpha
            )
            self.b2mc = MLSADigitalFilterCoefficientsToMelCepstrum(
                filter_order, alpha=alpha
            )

        self.stft = ShortTermFourierTransform(
            frame_length, frame_period, fft_length, out_format="complex", **stft_kwargs
        )
        self.istft = InverseShortTermFourierTransform(
            frame_length, frame_period, fft_length, **stft_kwargs
        )
        self.mgc2sp = MelGeneralizedCepstrumToSpectrum(
            filter_order,
            fft_length,
            alpha=alpha,
            gamma=gamma,
            out_format="magnitude" if phase == "zero" else "complex",
            n_fft=n_fft,
        )

    def forward(self, x, mc):
        if self.ignore_gain:
            b = self.mc2b(mc)
            b = self.gnorm(b)
            b[..., 0] = 0
            mc = self.b2mc(b)

        H = self.mgc2sp(mc)
        X = self.stft(x)
        Y = H * X
        y = self.istft(Y, out_length=x.size(-1))
        return y
