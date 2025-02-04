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
import torch.nn.functional as F
from torch import nn

from ..misc.utils import Lambda
from ..misc.utils import check_size
from ..misc.utils import get_gamma
from ..misc.utils import remove_gain
from .b2mc import MLSADigitalFilterCoefficientsToMelCepstrum
from .c2mpir import CepstrumToMinimumPhaseImpulseResponse
from .gnorm import GeneralizedCepstrumGainNormalization
from .istft import InverseShortTimeFourierTransform
from .linear_intpl import LinearInterpolation
from .mc2b import MelCepstrumToMLSADigitalFilterCoefficients
from .mgc2mgc import MelGeneralizedCepstrumToMelGeneralizedCepstrum
from .mgc2sp import MelGeneralizedCepstrumToSpectrum
from .stft import ShortTimeFourierTransform


def is_array_like(x):
    """Return True if the input is array-like.

    Parameters
    ----------
    x : object
        Any object.

    Returns
    -------
    out : bool
        True if the input is array-like.

    """
    return isinstance(x, (tuple, list))


def mirror(x, half=False):
    """Mirror the input tensor.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        Input tensor.

    half : bool
        If True, multiply all elements except the first one by 0.5.

    Returns
    -------
    out : Tensor [shape=(..., 2L-1)]
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
    filter_order : int >= 0 or tuple[int, int]
        Order of filter coefficients, :math:`M` or :math:`(N, M)`. A tuple input is
        allowed only if **phase** is 'mixed'.

    frame_period : int >= 1
        Frame period, :math:`P`.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    gamma : float in [-1, 1]
        Gamma, :math:`\\gamma`.

    c : int >= 1 or None
        Number of stages.

    ignore_gain : bool
        If True, perform filtering without gain.

    phase : ['minimum', 'maximum', 'zero', 'mixed']
        Filter type.

    mode : ['multi-stage', 'single-stage', 'freq-domain']
        'multi-stage' approximates the MLSA filter by cascading FIR filters based on the
        Taylor series expansion. 'single-stage' uses a FIR filter whose coefficients are
        the impulse response converted from input mel-cepstral coefficients using FFT.
        'freq-domain' performs filtering in the frequency domain rather than time one.

    n_fft : int >= 1
        Number of FFT bins used for conversion. Higher values result in increased
        conversion accuracy.

    taylor_order : int >= 0
        Order of Taylor series expansion (valid only if **mode** is 'multi-stage').

    cep_order : int >= 0 or tuple[int, int]
        Order of linear cepstrum (valid only if **mode** is 'multi-stage').

    ir_length : int >= 1 or tuple[int, int]
        Length of impulse response (valid only if **mode** is 'single-stage').

    **kwargs : additional keyword arguments
        See :func:`~diffsptk.ShortTimeFourierTransform` (valid only if **mode** is
        'freq-domain').

    References
    ----------
    .. [1] T. Yoshimura et al., "Embedding a differentiable mel-cepstral synthesis
           filter to a neural speech synthesis system," *Proceedings of ICASSP*, 2023.

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
        super().__init__()

        self.frame_period = frame_period

        # Format parameters.
        if phase == "mixed" and not is_array_like(filter_order):
            filter_order = (filter_order, filter_order)
        gamma = get_gamma(gamma, c)

        if phase == "mixed":
            self.split_sections = (filter_order[0], filter_order[1] + 1)
        else:
            self.split_sections = (filter_order + 1,)

        def flip(x):
            if is_array_like(x):
                return x[1], x[0]
            return x

        flip_keys = ("cep_order", "ir_length")
        modified_kwargs = kwargs.copy()
        for key in flip_keys:
            if key in kwargs:
                modified_kwargs[key] = flip(kwargs[key])
        flipped_filter_order = flip(filter_order)

        if mode == "multi-stage":
            self.mglsadf = MultiStageFIRFilter(
                flipped_filter_order,
                frame_period,
                alpha=alpha,
                gamma=gamma,
                ignore_gain=ignore_gain,
                phase=phase,
                **modified_kwargs,
            )
        elif mode == "single-stage":
            self.mglsadf = SingleStageFIRFilter(
                flipped_filter_order,
                frame_period,
                alpha=alpha,
                gamma=gamma,
                ignore_gain=ignore_gain,
                phase=phase,
                **modified_kwargs,
            )
        elif mode == "freq-domain":
            self.mglsadf = FrequencyDomainFIRFilter(
                flipped_filter_order,
                frame_period,
                alpha=alpha,
                gamma=gamma,
                ignore_gain=ignore_gain,
                phase=phase,
                **modified_kwargs,
            )
        else:
            raise ValueError(f"mode {mode} is not supported.")

    def forward(self, x, mc):
        """Apply an MGLSA digital filter.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Excitation signal.

        mc : Tensor [shape=(..., T/P, M+1)] or [shape=(..., T/P, N+M+1)]
            Mel-generalized cepstrum, not MLSA digital filter coefficients. Note that
            the mixed-phase case assumes that the coefficients are of the form
            c_{-N}, ..., c_{0}, ..., c_{M}, where M is the order of the minimum-phase
            part and N is the order of the maximum-phase part.

        Returns
        -------
        out : Tensor [shape=(..., T)]
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
        check_size(mc.size(-1), sum(self.split_sections), "dimension of mel-cepstrum")
        check_size(x.size(-1), mc.size(-2) * self.frame_period, "sequence length")
        if len(self.split_sections) != 1:
            mc_max, mc_min = torch.split(mc, self.split_sections, dim=-1)
            mc_max = F.pad(mc_max.flip(-1), (1, 0))
            mc = (mc_min, mc_max)  # (c0, c1, ..., cM), (0, c-1, ..., c-N)
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
        n_fft=512,
    ):
        super().__init__()

        assert 0 <= taylor_order

        self.ignore_gain = ignore_gain
        self.phase = phase
        self.taylor_order = taylor_order

        if alpha == 0 and gamma == 0:
            cep_order = filter_order

        # Prepare padding module.
        if self.phase == "minimum":
            padding = (cep_order, 0)
        elif self.phase == "maximum":
            padding = (0, cep_order)
        elif self.phase == "zero":
            padding = (cep_order, cep_order)
        elif self.phase == "mixed":
            padding = cep_order if is_array_like(cep_order) else (cep_order, cep_order)
        else:
            raise ValueError(f"phase {phase} is not supported.")
        self.pad = nn.ConstantPad1d(padding, 0)

        # Prepare frequency transformation module.
        if self.phase == "mixed":
            self.mgc2c = nn.ModuleList()
            for i in range(2):
                self.mgc2c.append(
                    MelGeneralizedCepstrumToMelGeneralizedCepstrum(
                        filter_order[i],
                        padding[i],
                        in_alpha=alpha,
                        in_gamma=gamma,
                        n_fft=n_fft,
                    )
                )
        else:
            self.mgc2c = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
                filter_order,
                cep_order,
                in_alpha=alpha,
                in_gamma=gamma,
                n_fft=n_fft,
            )

        self.linear_intpl = LinearInterpolation(frame_period)

    def forward(self, x, mc):
        if self.phase == "mixed":
            mc_min, mc_max = mc
            c_min = self.mgc2c[0](mc_min)
            c_max = self.mgc2c[1](mc_max)
            c0 = c_min[..., :1] + c_max[..., :1]
            c1_min = c_min[..., 1:].flip(-1)
            c0_dummy = torch.zeros_like(c0)
            c1_max = c_max[..., 1:]
            c = torch.cat([c1_min, c0_dummy, c1_max], dim=-1)
        else:
            c = self.mgc2c(mc)
            c0, c = remove_gain(c, value=0, return_gain=True)
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

        if not self.ignore_gain:
            K = torch.exp(self.linear_intpl(c0))
            y *= K.squeeze(-1)
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
        super().__init__()

        self.ignore_gain = ignore_gain
        self.phase = phase
        self.n_fft = n_fft

        # Prepare padding module.
        taps = ir_length - 1
        if self.phase == "minimum":
            padding = (taps, 0)
        elif self.phase == "maximum":
            padding = (0, taps)
        elif self.phase == "zero":
            padding = (taps, taps)
        elif self.phase == "mixed":
            padding = (
                (ir_length[0] - 1, ir_length[1] - 1)
                if is_array_like(ir_length)
                else (taps, taps)
            )
        else:
            raise ValueError(f"phase {phase} is not supported.")
        self.pad = nn.ConstantPad1d(padding, 0)
        self.padding = padding

        if self.phase in ("minimum", "maximum"):
            self.mgc2ir = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
                filter_order,
                ir_length - 1,
                in_alpha=alpha,
                in_gamma=gamma,
                out_gamma=1,
                out_mul=True,
                n_fft=n_fft,
            )
        elif self.phase == "zero":
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
        elif self.phase == "mixed":
            self.mgc2c = nn.ModuleList()
            for i in range(2):
                self.mgc2c.append(
                    MelGeneralizedCepstrumToMelGeneralizedCepstrum(
                        filter_order[i],
                        padding[i],
                        in_alpha=alpha,
                        in_gamma=gamma,
                        n_fft=n_fft,
                    )
                )
            self.c2ir = CepstrumToMinimumPhaseImpulseResponse(
                n_fft - 1, n_fft, n_fft=n_fft
            )
        else:
            raise ValueError(f"phase {phase} is not supported.")

        self.linear_intpl = LinearInterpolation(frame_period)

    def forward(self, x, mc):
        if self.phase == "minimum":
            h = self.mgc2ir(mc)
            h = h.flip(-1)
        elif self.phase == "maximum":
            h = self.mgc2ir(mc)
        elif self.phase == "zero":
            c = self.mgc2c(mc)
            c[..., 1:] *= 0.5
            if self.ignore_gain:
                c = remove_gain(c, value=0)
            h = self.c2ir(c)
            h = mirror(h)
        elif self.phase == "mixed":
            mc_min, mc_max = mc
            c_min = self.mgc2c[0](mc_min)
            c_max = self.mgc2c[1](mc_max)
            if self.ignore_gain:
                c0 = torch.zeros_like(c_min[..., :1])
            else:
                c0 = c_min[..., :1] + c_max[..., :1]
            c = torch.cat([c_min[..., 1:].flip(-1), c0, c_max[..., 1:]], dim=-1)
            c = F.pad(c, (0, self.n_fft - c.size(-1)))
            c = torch.roll(c, -self.padding[0], dims=-1)
            h = self.c2ir(c)
            h = torch.roll(h, self.padding[0], dims=-1)[..., : sum(self.padding) + 1]
        else:
            raise RuntimeError

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
        super().__init__()

        assert 2 * frame_period < frame_length

        self.ignore_gain = ignore_gain
        self.phase = phase

        if self.ignore_gain:
            self.gnorm = nn.ModuleList()
            self.mc2b = nn.ModuleList()
            self.b2mc = nn.ModuleList()
        self.mgc2sp = nn.ModuleList()

        if not is_array_like(filter_order):
            filter_order = (filter_order, filter_order)

        n = 2 if phase == "mixed" else 1
        for i in range(n):
            if self.ignore_gain:
                self.gnorm.append(
                    GeneralizedCepstrumGainNormalization(filter_order[i], gamma=gamma)
                )
                self.mc2b.append(
                    MelCepstrumToMLSADigitalFilterCoefficients(
                        filter_order[i], alpha=alpha
                    )
                )
                self.b2mc.append(
                    MLSADigitalFilterCoefficientsToMelCepstrum(
                        filter_order[i], alpha=alpha
                    )
                )
            self.mgc2sp.append(
                MelGeneralizedCepstrumToSpectrum(
                    filter_order[i],
                    fft_length,
                    alpha=alpha,
                    gamma=gamma,
                    out_format="complex",
                    n_fft=n_fft,
                )
            )

        self.stft = ShortTimeFourierTransform(
            frame_length, frame_period, fft_length, out_format="complex", **stft_kwargs
        )
        self.istft = InverseShortTimeFourierTransform(
            frame_length, frame_period, fft_length, **stft_kwargs
        )

    def forward(self, x, mc):
        if torch.is_tensor(mc):
            mc = [mc]

        Hs = []
        for i, c in enumerate(mc):
            if self.ignore_gain:
                b = self.mc2b[i](c)
                b = self.gnorm[i](b)
                b[..., 0] = 0
                c = self.b2mc[i](b)
            Hs.append(self.mgc2sp[i](c))

        if self.phase == "minimum":
            H = Hs[0]
        elif self.phase == "maximum":
            H = Hs[0].conj()
        elif self.phase == "zero":
            H = Hs[0].abs()
        elif self.phase == "mixed":
            H = Hs[0] * Hs[1].conj()
        else:
            raise RuntimeError

        X = self.stft(x)
        Y = H * X
        y = self.istft(Y, out_length=x.size(-1))
        return y
