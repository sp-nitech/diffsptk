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

from typing import Any

import mpmath as mp
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..utils.private import Lambda, check_size, get_gamma, remove_gain, to
from .b2mc import MLSADigitalFilterCoefficientsToMelCepstrum
from .base import BaseNonFunctionalModule
from .c2mpir import CepstrumToMinimumPhaseImpulseResponse
from .frame import Frame
from .gnorm import GeneralizedCepstrumGainNormalization
from .istft import InverseShortTimeFourierTransform
from .linear_intpl import LinearInterpolation
from .mc2b import MelCepstrumToMLSADigitalFilterCoefficients
from .mgc2mgc import MelGeneralizedCepstrumToMelGeneralizedCepstrum
from .mgc2sp import MelGeneralizedCepstrumToSpectrum
from .root_pol import PolynomialToRoots
from .stft import ShortTimeFourierTransform


def is_array_like(x: Any) -> bool:
    return isinstance(x, (tuple, list, np.ndarray))


def mirror(x: torch.Tensor, half: bool = False) -> torch.Tensor:
    x0, x1 = torch.split(x, [1, x.size(-1) - 1], dim=-1)
    if half:
        x1 = x1 * 0.5
    y = torch.cat((x1.flip(-1), x0, x1), dim=-1)
    return y


class PseudoMGLSADigitalFilter(BaseNonFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mglsadf.html>`_
    for details.

    Parameters
    ----------
    filter_order : int >= 0 or tuple[int, int]
        The order of the filter coefficients, :math:`M` or :math:`(N, M)`. A tuple input
        is allowed only if **phase** is 'mixed'.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    gamma : float in [-1, 1]
        The gamma parameter, :math:`\\gamma`.

    c : int >= 1 or None
        The number of stages.

    ignore_gain : bool
        If True, filtering is performed without gain.

    phase : ['minimum', 'maximum', 'zero', 'mixed']
        The filter type.

    mode : ['multi-stage', 'single-stage', 'freq-domain', 'pade-approx']
        'multi-stage' approximates the MLSA filter by cascading FIR filters based on the
        Taylor series expansion. 'single-stage' uses an FIR filter with the coefficients
        derived from the impulse response converted from the input mel-cepstral
        coefficients using FFT. 'freq-domain' performs filtering in the frequency domain
        rather than the time domain. 'pade-approx' implements the MLSA filter by
        cascading all-zero and all-pole filters based on the factorization. This is slow,
        but can be used to optimize the Pade approximation coefficients.

    n_fft : int >= 1
        The number of FFT bins used for conversion. Higher values result in increased
        conversion accuracy.

    taylor_order : int >= 0
        The order of the Taylor series expansion (valid only if **mode** is
        'multi-stage').

    pade_order : int >= 3
        The order of Pade approximation (valid only if **mode** is 'pade-approx').

    cep_order : int >= 0 or tuple[int, int]
        The order of the linear cepstrum (valid only if **mode** is 'multi-stage' or
        'pade-approx').

    ir_length : int >= 1 or tuple[int, int]
        The length of the impulse response (valid only if **mode** is 'single-stage').

    learnable : bool
        If True, the polynomial coefficients used in the approximation are learnable
        (valid only if **mode** is 'multi-stage' or 'pade-approx').

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

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
        filter_order: tuple[int, int] | int,
        frame_period: int,
        *,
        alpha: float = 0,
        gamma: float = 0,
        c: int | None = None,
        ignore_gain: bool = False,
        phase: str = "minimum",
        mode: str = "multi-stage",
        **kwargs,
    ) -> None:
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
        elif mode == "pade-approx":
            self.mglsadf = MultiStageIIRFilter(
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

    def forward(self, x: torch.Tensor, mc: torch.Tensor) -> torch.Tensor:
        """Apply an MGLSA digital filter.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The excitation signal.

        mc : Tensor [shape=(..., T/P, M+1)] or [shape=(..., T/P, N+M+1)]
            The mel-generalized cepstrum, not MLSA digital filter coefficients. Note
            that the mixed-phase case assumes that the coefficients are of the form
            c_{-N}, ..., c_{0}, ..., c_{M}, where M is the order of the minimum-phase
            part and N is the order of the maximum-phase part.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The output signal.

        Examples
        --------
        >>> import diffsptk
        >>> import torch
        >>> mglsadf = diffsptk.MLSA(1, frame_period=2)
        >>> y = diffsptk.step(3)
        >>> mc = torch.tensor([[0.3, 0.5], [-0.2, 0.1]])
        >>> x = mglsadf(y, mc)
        >>> x
        tensor([1.3499, 1.3667, 0.9129, 0.9051])

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
        filter_order: tuple[int, int] | int,
        frame_period: int,
        *,
        alpha: float = 0,
        gamma: float = 0,
        ignore_gain: bool = False,
        phase: str = "minimum",
        taylor_order: int = 20,
        cep_order: tuple[int, int] | int = 199,
        n_fft: int = 512,
        learnable: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if taylor_order < 0:
            raise ValueError("taylor_order must be non-negative.")

        self.ignore_gain = ignore_gain
        self.phase = phase

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
                        device=device,
                        dtype=dtype,
                    )
                )
        else:
            self.mgc2c = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
                filter_order,
                cep_order,
                in_alpha=alpha,
                in_gamma=gamma,
                n_fft=n_fft,
                device=device,
                dtype=dtype,
            )

        self.linear_intpl = LinearInterpolation(frame_period)

        cp = mp.taylor(mp.exp, 0, taylor_order)
        cp = np.array([float(x) for x in cp])
        weights = cp[1:] / cp[:-1]
        weights = np.insert(weights, 0, 1)
        self.register_buffer("weights", to(weights, device=device, dtype=dtype))

        a = np.ones(taylor_order + 1)
        a = to(a, device=device, dtype=dtype)
        if learnable:
            self.a = nn.Parameter(a)
        else:
            self.register_buffer("a", a)

    def forward(
        self,
        x: torch.Tensor,
        mc: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
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

        y = x * self.a[0]
        for i in range(1, len(self.a)):
            x = self.pad(x)
            x = x.unfold(-1, c.size(-1), 1)
            x = (x * c).sum(-1) * self.weights[i]
            y += x * self.a[i]

        if not self.ignore_gain:
            K = torch.exp(self.linear_intpl(c0))
            y *= K.squeeze(-1)
        return y


class SingleStageFIRFilter(nn.Module):
    def __init__(
        self,
        filter_order: tuple[int, int] | int,
        frame_period: int,
        *,
        alpha: float = 0,
        gamma: float = 0,
        ignore_gain: bool = False,
        phase: str = "minimum",
        ir_length: tuple[int, int] | int = 2000,
        n_fft: int = 4096,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
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
                device=device,
                dtype=dtype,
            )
        elif self.phase == "zero":
            self.mgc2c = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
                filter_order,
                ir_length - 1,
                in_alpha=alpha,
                in_gamma=gamma,
                n_fft=n_fft,
                device=device,
                dtype=dtype,
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
                        device=device,
                        dtype=dtype,
                    )
                )
            self.c2ir = CepstrumToMinimumPhaseImpulseResponse(
                n_fft - 1, n_fft, n_fft=n_fft
            )
        else:
            raise ValueError(f"phase {phase} is not supported.")

        self.linear_intpl = LinearInterpolation(frame_period)

    def forward(
        self,
        x: torch.Tensor,
        mc: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
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
        filter_order: tuple[int, int] | int,
        frame_period: int,
        *,
        alpha: float = 0,
        gamma: float = 0,
        ignore_gain: bool = False,
        phase: str = "minimum",
        frame_length: int = 400,
        fft_length: int = 512,
        n_fft: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **stft_kwargs,
    ) -> None:
        super().__init__()

        if frame_length <= 2 * frame_period:
            raise ValueError("frame_period must be less than half of frame_length.")

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
                        filter_order[i], alpha=alpha, device=device, dtype=dtype
                    )
                )
                self.b2mc.append(
                    MLSADigitalFilterCoefficientsToMelCepstrum(
                        filter_order[i],
                        alpha=alpha,
                        device=device,
                        dtype=dtype,
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
                    device=device,
                    dtype=dtype,
                )
            )

        self.stft = ShortTimeFourierTransform(
            frame_length,
            frame_period,
            fft_length,
            out_format="complex",
            device=device,
            dtype=dtype,
            **stft_kwargs,
        )
        self.istft = InverseShortTimeFourierTransform(
            frame_length,
            frame_period,
            fft_length,
            device=device,
            dtype=dtype,
            **stft_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        mc: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
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


class MultiStageIIRFilter(nn.Module):
    def __init__(
        self,
        filter_order: tuple[int, int] | int,
        frame_period: int,
        *,
        alpha: float = 0,
        gamma: float = 0,
        ignore_gain: bool = False,
        phase: str = "minimum",
        pade_order: int = 5,
        cep_order: tuple[int, int] | int = 199,
        n_fft: int = 512,
        chunk_length: int | None = None,
        warmup_length: int | None = None,
        learnable: bool = False,
        per_stage_pade_coefficients: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if phase != "minimum" or is_array_like(filter_order):
            raise ValueError("Only minimum-phase filter is supported.")

        self.ignore_gain = ignore_gain

        self.mgc2c = MelGeneralizedCepstrumToMelGeneralizedCepstrum(
            filter_order,
            cep_order,
            in_alpha=alpha,
            in_gamma=gamma,
            n_fft=n_fft,
            device=device,
            dtype=dtype,
        )
        self.linear_intpl = LinearInterpolation(frame_period)
        self.root_pol = PolynomialToRoots(pade_order, device=device, dtype=dtype)

        from torchlpc import sample_wise_lpc

        self.sample_wise_lpc = sample_wise_lpc

        if chunk_length is None:
            self.chuking = False
        else:
            self.chuking = True
            self.warmup_length = (
                warmup_length if warmup_length is not None else cep_order
            )
            if chunk_length <= 0:
                raise ValueError("chunk_length must be positive.")
            if self.warmup_length < 0:
                raise ValueError("warmup_length must be non-negative.")
            frame_period = chunk_length - self.warmup_length
            self.frame_x = Frame(chunk_length, frame_period, center=False)
            self.frame_c = Frame(
                cep_order * chunk_length, cep_order * frame_period, center=False
            )

        cr = mp.taylor(mp.exp, 0, pade_order * 2)
        cp, cq = mp.pade(cr, pade_order, pade_order)
        cp = np.array([float(x) for x in cp])
        weights = cp[1:] / cp[:-1]
        weights = np.insert(weights, 0, 1)
        self.register_buffer("weights", to(weights, device=device, dtype=dtype))

        if pade_order == 3:
            a1 = np.linspace(1.0, 0.4, pade_order + 1)
        elif pade_order == 4:
            a1 = np.linspace(1.0, 0.6, pade_order + 1)
        elif 5 <= pade_order <= 14:
            a1 = np.ones(pade_order + 1)
        else:
            raise ValueError("pade_order must be in [3, 14].")

        if per_stage_pade_coefficients:
            a2 = a1
            a1 = np.ones(pade_order + 1)
            a1 = to(a1, device=device, dtype=dtype)
            a2 = to(a2, device=device, dtype=dtype)
            if learnable:
                self.a1 = nn.Parameter(a1)
                self.a2 = nn.Parameter(a2)
            else:
                self.register_buffer("a1", a1)
                self.register_buffer("a2", a2)
        else:
            a1 = to(a1, device=device, dtype=dtype)
            if learnable:
                self.a1 = nn.Parameter(a1)
            else:
                self.register_buffer("a1", a1)
            self.a2 = self.a1

    def forward(
        self, x: torch.Tensor, mc: torch.Tensor, return_roots: bool = False
    ) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
            mc = mc.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        if x.dim() != 2 or mc.dim() != 3:
            raise ValueError("x and mc must be 2-D and 3-D tensors, respectively.")

        c = self.mgc2c(mc)
        c0, c1 = torch.split(c, [1, c.size(-1) - 1], dim=-1)
        c_b = self.linear_intpl(c1.flip(-1))
        c_a = self.linear_intpl(c1)

        T = x.size(-1)
        B, _, M = c_a.size()

        a1 = torch.clip(self.a1, min=1e-1, max=1e+1)
        a1[0] = 1.0
        a2 = torch.clip(self.a2, min=1e-1, max=1e+1)
        a2[0] = 1.0

        c_b2, c_b1 = torch.split(c_b, [c_b.size(-1) - 1, 1], dim=-1)
        c_b1 = c_b1.squeeze(-1)

        # Numerator, 1st stage:
        y = x * a1[0]
        for i in range(1, len(a1)):
            x = F.pad(x[..., :-1], (1, 0))
            x = x * c_b1 * self.weights[i]
            y += x * a1[i]

        # Numerator, 2nd stage:
        x = y
        y = x * a2[0]
        for i in range(1, len(a2)):
            x = F.pad(x, (M, 0))
            x = x.unfold(-1, M + 1, 1)
            x = (x[..., :-2] * c_b2).sum(-1) * self.weights[i]
            y += x * a2[i]

        if self.chuking:
            y = F.pad(y, (self.warmup_length, 0))
            y = self.frame_x(y)
            y = y.reshape(-1, y.size(-1))

            c_a = c_a.reshape(B, -1)
            c_a = F.pad(c_a, (M * self.warmup_length, 0))
            c_a = self.frame_c(c_a)
            c_a = c_a.reshape(y.size(0), y.size(1), M)

        c_a1, c_a2 = torch.split(c_a, [1, c_a.size(-1) - 1], dim=-1)
        c_a2 = F.pad(c_a2, (1, 0))

        def compute_roots(a: torch.Tensor) -> torch.Tensor:
            pade_coefficients = torch.cumprod(self.weights, 0) * a
            roots = self.root_pol(pade_coefficients.flip(0).double())
            roots = roots.to(
                torch.complex64 if a.dtype == torch.float32 else torch.complex128
            )
            return roots

        roots1 = compute_roots(a1)
        roots2 = compute_roots(a2)
        roots = torch.stack([roots1, roots2], dim=0)

        # Denominator, 1st stage:
        y = y.to(roots.dtype)
        p1 = torch.reciprocal(roots1)
        for i in range(len(p1)):
            y = self.sample_wise_lpc(y, (p1[i] * c_a1))

        # Denominator, 2nd stage:
        p2 = torch.reciprocal(roots2)
        for i in range(len(p2)):
            y = self.sample_wise_lpc(y, (p2[i] * c_a2))
        y = y.real

        if self.chuking:
            y = y[..., self.warmup_length :]
            y = y.reshape(B, -1)
            y = y[..., :T]

        if not self.ignore_gain:
            K = torch.exp(self.linear_intpl(c0))
            y = y * K.squeeze(-1)

        if unsqueezed:
            y = y.squeeze(0)

        if return_roots:
            return y, roots
        return y
