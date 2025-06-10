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

from torch import Tensor

from . import modules as nn
from .typing import ArrayLike


def acorr(x: Tensor, acr_order: int, out_format: str = "naive") -> Tensor:
    """Estimate the autocorrelation of the input waveform.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        The framed waveform.

    acr_order : int >= 0
        The order of the autocorrelation, :math:`M`.

    out_format : ['naive', 'normalized', 'biased']
        The type of the autocorrelation.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The autocorrelation.

    """
    return nn.Autocorrelation._func(x, acr_order=acr_order, out_format=out_format)


def acr2csm(r: Tensor) -> Tensor:
    """Convert autocorrelation to CSM coefficients.

    Parameters
    ----------
    r : Tensor [shape=(..., M+1)]
        The autocorrelation.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The CSM coefficients.

    """
    return nn.AutocorrelationToCompositeSinusoidalModelCoefficients._func(r)


def alaw(x: Tensor, abs_max: float = 1, a: float = 87.6) -> Tensor:
    """Compress the input waveform using the A-law algorithm.

    Parameters
    ----------
    x : Tensor [shape=(...,)]
        The input waveform.

    abs_max : float > 0
        The absolute maximum value of the input waveform.

    a : float >= 1
        The compression factor, :math:`A`.

    Returns
    -------
    out : Tensor [shape=(...,)]
        The compressed waveform.

    """
    return nn.ALawCompression._func(x, abs_max=abs_max, a=a)


def b2mc(b: Tensor, alpha: float = 0) -> Tensor:
    """Convert MLSA filter coefficients to mel-cepstrum.

    Parameters
    ----------
    b : Tensor [shape=(..., M+1)]
        The MLSA filter coefficients.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The mel-cepstral coefficients.

    """
    return nn.MLSADigitalFilterCoefficientsToMelCepstrum._func(b, alpha=alpha)


def c2acr(c: Tensor, acr_order: int, n_fft: int = 512) -> Tensor:
    """Convert cepstrum to autocorrelation.

    Parameters
    ----------
    c : Tensor [shape=(..., M+1)]
        The cepstral coefficients.

    acr_order : int >= 0
        The order of the autocorrelation, :math:`N`.

    n_fft : int >> N
        The number of FFT bins used for conversion.

    Returns
    -------
    out : Tensor [shape=(..., N+1)]
        The autocorrelation.

    """
    return nn.CepstrumToAutocorrelation._func(c, acr_order=acr_order, n_fft=n_fft)


def c2mpir(c: Tensor, ir_length: int, n_fft: int = 512) -> Tensor:
    """Convert cepstrum to minimum phase impulse response.

    Parameters
    ----------
    c : Tensor [shape=(..., M+1)]
        The cepstral coefficients.

    ir_length : int >= 1
        The length of the impulse response, :math:`N`.

    n_fft : int >> N
        The number of FFT bins used for conversion.

    Returns
    -------
    out : Tensor [shape=(..., N)]
        The truncated minimum phase impulse response.

    """
    return nn.CepstrumToMinimumPhaseImpulseResponse._func(
        c, ir_length=ir_length, n_fft=n_fft
    )


def c2ndps(c: Tensor, fft_length: int) -> Tensor:
    """Convert cepstrum to NDPS.

    Parameters
    ----------
    c : Tensor [shape=(..., M+1)]
        The cepstrum.

    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    Returns
    -------
    out : Tensor [shape=(..., L/2+1)]
        The NDPS.

    """
    return nn.CepstrumToNegativeDerivativeOfPhaseSpectrum._func(
        c, fft_length=fft_length
    )


def cdist(
    c1: Tensor, c2: Tensor, full: bool = False, reduction: str = "mean"
) -> Tensor:
    """Calculate the cepstral distance between two inputs.

    Parameters
    ----------
    c1 : Tensor [shape=(..., M+1)]
        The input cepstral coefficients.

    c2 : Tensor [shape=(..., M+1)]
        The target cepstral coefficients.

    full : bool
        If True, include the constant term in the distance calculation.

    reduction : ['none', 'mean', 'batchmean', 'sum']
        The reduction type.

    Returns
    -------
    out : Tensor [shape=(...,) or scalar]
        The cepstral distance.

    """
    return nn.CepstralDistance._func(c1, c2, full=full, reduction=reduction)


def chroma(
    x: Tensor,
    n_channel: int,
    sample_rate: int,
    norm: float = float("inf"),
    use_power: bool = True,
) -> Tensor:
    """Apply chroma-filter banks to the STFT.

    Parameters
    ----------
    x : Tensor [shape=(..., L/2+1)]
        The power spectrum.

    n_channel : int >= 1
        The number of chroma filter banks, :math:`C`.

    sample_rate : int >= 1
        The sample rate in Hz.

    norm : float
        The normalization factor.

    use_power : bool
        If True, use the power spectrum instead of the amplitude spectrum.

    Returns
    -------
    out : Tensor [shape=(..., C)]
        The chroma filter bank output.

    """
    return nn.ChromaFilterBankAnalysis._func(
        x,
        n_channel=n_channel,
        sample_rate=sample_rate,
        norm=norm,
        use_power=use_power,
    )


def csm2acr(c: Tensor) -> Tensor:
    """Convert CSM coefficients to autocorrelation.

    Parameters
    ----------
    c : Tensor [shape=(..., M+1)]
        The CSM coefficients.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The autocorrelation.

    """
    return nn.CompositeSinusoidalModelCoefficientsToAutocorrelation._func(c)


def dct(x: Tensor, dct_type: int = 2) -> Tensor:
    """Compute DCT.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        The input.

    dct_type : int in [1, 4]
        The DCT type.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        The DCT output.

    """
    return nn.DiscreteCosineTransform._func(x, dct_type=dct_type)


def decimate(x: Tensor, period: int = 1, start: int = 0, dim: int = -1) -> Tensor:
    """Decimate the input signal.

    Parameters
    ----------
    x : Tensor [shape=(..., T, ...)]
        The input signal.

    period : int >= 1
        The decimation period, :math:`P`.

    start : int >= 0
        The start point, :math:`S`.

    dim : int
        The dimension along which to decimate the tensors.

    Returns
    -------
    out : Tensor [shape=(..., T/P-S, ...)]
        The decimated signal.

    """
    return nn.Decimation._func(x, period=period, start=start, dim=dim)


def delay(x: Tensor, start: int = 0, keeplen: bool = False, dim: int = -1) -> Tensor:
    """Delay the input signal.

    Parameters
    ----------
    x : Tensor [shape=(..., T, ...)]
        The input signal.

    start : int
        The start point, :math:`S`. If negative, advance the signal.

    keeplen : bool
        If True, the output has the same length of the input.

    dim : int
        The dimension along which to delay the tensors.

    Returns
    -------
    out : Tensor [shape=(..., T-S, ...)] or [shape=(..., T, ...)]
        The delayed signal.

    """
    return nn.Delay._func(x, start=start, keeplen=keeplen, dim=dim)


def delta(
    x: Tensor,
    seed: ArrayLike[ArrayLike[float]] | ArrayLike[int] = [[-0.5, 0, 0.5]],
    static_out: bool = True,
) -> Tensor:
    """Compute the delta components.

    Parameters
    ----------
    x : Tensor [shape=(B, T, D) or (T, D)]
        The static components.

    seed : list[list[float]] or list[int]
        The delta coefficients or the width(s) of 1st (and 2nd) regression coefficients.

    static_out : bool
        If False, outputs only the delta components.

    Returns
    -------
    out : Tensor [shape=(B, T, DxH) or (T, DxH)]
        The delta (and static) components.

    """
    return nn.Delta._func(x, seed, static_out=static_out)


def dequantize(
    y: Tensor, abs_max: float = 1, n_bit: int = 8, quantizer: str = "mid-rise"
) -> Tensor:
    """Dequantize the input waveform.

    Parameters
    ----------
    y : Tensor [shape=(...,)]
        The quantized waveform.

    abs_max : float > 0
        The absolute maximum value of the original waveform.

    n_bit : int >= 1
        The number of quantization bits.

    quantizer : ['mid-rise', 'mid-tread']
        The quantizer type.

    Returns
    -------
    out : Tensor [shape=(...,)]
        The dequantized waveform.

    """
    return nn.InverseUniformQuantization._func(
        y, abs_max=abs_max, n_bit=n_bit, quantizer=quantizer
    )


def dfs(
    x: Tensor,
    b: Tensor | None = None,
    a: Tensor | None = None,
    ir_length: int | None = None,
) -> Tensor:
    """Apply an IIR digital filter to the input waveform.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        The input waveform.

    b : Tensor [shape=(M+1,)] or None
        The numerator coefficients.

    a : Tensor [shape=(N+1,)] or None
        The denominator coefficients.

    ir_length : int >= 1 or None
        The length of the truncated impulse response. If given, the filter is
        approximated by an FIR filter.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        The filtered waveform.

    """
    return nn.InfiniteImpulseResponseDigitalFilter._func(
        x, b=b, a=a, ir_length=ir_length
    )


def dht(x: Tensor, dht_type: int = 2) -> Tensor:
    """Compute DHT.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        The input.

    dht_type : int in [1, 4]
        The DHT type.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        The DHT output.

    """
    return nn.DiscreteHartleyTransform._func(x, dht_type=dht_type)


def drc(
    x: Tensor,
    threshold: float,
    ratio: float,
    attack_time: float,
    release_time: float,
    sample_rate: int,
    makeup_gain: float = 0,
    abs_max: float = 1,
) -> Tensor:
    """Perform dynamic range compression.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        The input waveform.

    threshold : float <= 0
        The threshold in dB.

    ratio : float > 1
        The input/output ratio.

    attack_time : float > 0
        The attack time in msec.

    release_time : float > 0
        The release time in msec.

    sample_rate : int >= 1
        The sample rate in Hz.

    makeup_gain : float >= 0
        The make-up gain in dB.

    abs_max : float > 0
        The absolute maximum value of input.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        The compressed waveform.

    """
    return nn.DynamicRangeCompression._func(
        x,
        threshold=threshold,
        ratio=ratio,
        attack_time=attack_time,
        release_time=release_time,
        sample_rate=sample_rate,
        makeup_gain=makeup_gain,
        abs_max=abs_max,
    )


def dst(x: Tensor, dst_type: int = 2) -> Tensor:
    """Compute DST.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        The input.

    dst_type : int in [1, 4]
        The DST type.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        The DST output.

    """
    return nn.DiscreteSineTransform._func(x, dst_type=dst_type)


def entropy(p: Tensor, out_format: str = "nat") -> Tensor:
    """Calculate the entropy of a probability distribution.

    Parameters
    ----------
    p : Tensor [shape=(..., N)]
        The probability.

    out_format : ['bit', 'nat', 'dit']
        The output format.

    Returns
    -------
    out : Tensor [shape=(...,)]
        The entropy.

    """
    return nn.Entropy._func(p, out_format=out_format)


def excite(
    p: Tensor,
    frame_period: int = 80,
    *,
    voiced_region: str = "pulse",
    unvoiced_region: str = "gauss",
    polarity: str = "auto",
    init_phase: str = "zeros",
) -> Tensor:
    """Generate a simple excitation signal.

    Parameters
    ----------
    p : Tensor [shape=(..., N)]
        The pitch in seconds.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    voiced_region : ['pulse', 'sinusoidal', 'sawtooth', 'inverted-sawtooth', \
                     'triangle', 'square']
        The type of voiced region.

    unvoiced_region : ['zeros', 'gauss']
        The type of unvoiced region.

    polarity : ['auto', 'unipolar', 'bipolar']
        The polarity.

    init_phase : ['zeros', 'random']
        The initial phase.

    Returns
    -------
    out : Tensor [shape=(..., NxP)]
        The excitation signal.

    """
    return nn.ExcitationGeneration._func(
        p,
        frame_period=frame_period,
        voiced_region=voiced_region,
        unvoiced_region=unvoiced_region,
        polarity=polarity,
        init_phase=init_phase,
    )


def fbank(
    x: Tensor,
    n_channel: int,
    sample_rate: int,
    f_min: float = 0,
    f_max: float | None = None,
    floor: float = 1e-5,
    gamma: float = 0,
    scale: str = "htk",
    erb_factor: float | None = None,
    use_power: bool = False,
    out_format: str = "y",
) -> tuple[Tensor, Tensor] | Tensor:
    """Apply mel-filter banks to the STFT.

    Parameters
    ----------
    x : Tensor [shape=(..., L/2+1)]
        The power spectrum.

    n_channel : int >= 1
        The number of mel filter banks, :math:`C`.

    sample_rate : int >= 1
        The sample rate in Hz.

    f_min : float >= 0
        The minimum frequency in Hz.

    f_max : float <= sample_rate // 2
        The maximum frequency in Hz.

    floor : float > 0
        The minimum mel filter bank output in linear scale.

    gamma : float in [-1, 1]
        The parameter of the generalized logarithmic function.

    scale : ['htk', 'mel', 'inverted-mel', 'bark', 'linear']
        The type of auditory scale used to construct the filter bank.

    erb_factor : float > 0 or None
        The scale factor for the ERB scale, referred to as the E-factor. If not None,
        the filter bandwidths are adjusted according to the scaled ERB scale.

    use_power : bool
        If True, use the power spectrum instead of the amplitude spectrum.

    out_format : ['y', 'yE', 'y,E']
        `y` is mel filber bank output and `E` is energy. If this is `yE`, the two output
        tensors are concatenated and return the tensor instead of the tuple.

    Returns
    -------
    y : Tensor [shape=(..., C)]
        The mel filter bank output.

    E : Tensor [shape=(..., 1)] (optional)
        The energy.

    """
    return nn.MelFilterBankAnalysis._func(
        x,
        n_channel=n_channel,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        floor=floor,
        gamma=gamma,
        scale=scale,
        erb_factor=erb_factor,
        use_power=use_power,
        out_format=out_format,
    )


def fftcep(x: Tensor, cep_order: int, accel: float = 0, n_iter: int = 0) -> Tensor:
    """Perform cepstral analysis.

    Parameters
    ----------
    x : Tensor [shape=(..., L/2+1)]
        The power spectrum.

    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    accel : float >= 0
        The acceleration factor.

    n_iter : int >= 0
        The number of iterations.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The cepstrum.

    """
    return nn.CepstralAnalysis._func(x, cep_order=cep_order, accel=accel, n_iter=n_iter)


def fftr(
    x: Tensor, fft_length: int | None = None, out_format: str = "complex"
) -> Tensor:
    """Compute FFT of a real signal.

    Parameters
    ----------
    x : Tensor [shape=(..., N)]
        The real input signal.

    fft_length : int >= 2 or None
        The FFT length, :math:`L`.

    out_format : ['complex', 'real', 'imaginary', 'amplitude', 'power']
        The output format.

    Returns
    -------
    out : Tensor [shape=(..., L/2+1)]
        The output spectrum.

    """
    return nn.RealValuedFastFourierTransform._func(
        x, fft_length=fft_length, out_format=out_format
    )


def flux(
    x: Tensor,
    y: Tensor | None = None,
    lag: int = 1,
    norm: int | float = 2,
    reduction: str = "mean",
) -> Tensor:
    """Calculate flux.

    Parameters
    ----------
    x : Tensor [shape=(..., N, D)]
        The input.

    y : Tensor [shape=(..., N, D)] or None
        The target (optional).

    lag : int or float
        The lag of the distance calculation, :math:`L`.

    norm : int or float
        The order of the norm.

    reduction : ['none', 'mean', 'batchmean', 'sum']
        The reduction type.

    Returns
    -------
    out : Tensor [shape=(..., N-\\|L\\|) or scalar]
        The flux.

    """
    return nn.Flux._func(x, y, lag=lag, norm=norm, reduction=reduction)


def frame(
    x: Tensor,
    frame_length: int = 400,
    frame_period: int = 80,
    center: bool = True,
    zmean: bool = False,
    mode: str = "constant",
) -> Tensor:
    """Apply framing to the given waveform.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        The waveform.

    frame_length : int >= 1
        The frame length in samples, :math:`L`.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    center : bool
        If True, pad the input on both sides so that the frame is centered.

    zmean : bool
        If True, perform mean subtraction on each frame.

    mode : ['constant', 'reflect', 'replicate', 'circular']
        The padding method.

    Returns
    -------
    out : Tensor [shape=(..., T/P, L)]
        The framed waveform.

    """
    return nn.Frame._func(
        x,
        frame_length=frame_length,
        frame_period=frame_period,
        center=center,
        zmean=zmean,
        mode=mode,
    )


def freqt(c: Tensor, out_order: int, alpha: float = 0) -> Tensor:
    """Perform frequency transform.

    Parameters
    ----------
    c : Tensor [shape=(..., M1+1)]
        The cepstral coefficients.

    out_order : int >= 0
        The order of the output cepstrum, :math:`M_2`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    Returns
    -------
    out : Tensor [shape=(..., M2+1)]
        The warped cepstral coefficients.

    """
    return nn.FrequencyTransform._func(c, out_order=out_order, alpha=alpha)


def freqt2(
    c: Tensor, out_order: int, alpha: float = 0, theta: float = 0, n_fft: int = 512
) -> Tensor:
    """Perform second-order all-pass frequency transform.

    Parameters
    ----------
    c : Tensor [shape=(..., M1+1)]
        The cepstral coefficients.

    out_order : int >= 0
        The order of the output sequence, :math:`M_2`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    theta : float in [0, 1]
        The emphasis frequency, :math:`\\theta`.

    n_fft : int >> M2
        The number of FFT bins. The accurate conversion requires the large value.

    Returns
    -------
    out : Tensor [shape=(..., M2+1)]
        The warped cepstral coefficients.

    """
    return nn.SecondOrderAllPassFrequencyTransform._func(
        c, out_order=out_order, alpha=alpha, theta=theta, n_fft=n_fft
    )


def gnorm(x: Tensor, gamma: float = 0, c: int | None = None) -> Tensor:
    """Perform cepstrum gain normalization.

    Parameters
    ----------
    x : Tensor [shape=(..., M+1)]
        The generalized cepstrum.

    gamma : float in [-1, 1]
        The gamma parameter, :math:`\\gamma`.

    c : int >= 1 or None
        The number of filter stages.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The normalized generalized cepstrum.

    """
    return nn.GeneralizedCepstrumGainNormalization._func(x, gamma=gamma, c=c)


def griffin(
    y: Tensor,
    *,
    out_length: int | None = None,
    frame_length: int = 400,
    frame_period: int = 80,
    fft_length: int = 512,
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
) -> Tensor:
    """Reconstruct a waveform from the spectrum using the Griffin-Lim algorithm.

    Parameters
    ----------
    y : Tensor [shape=(..., T/P, N/2+1)]
        The power spectrum.

    out_length : int > 0 or None
        The length of the output waveform.

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

    n_iter : int >= 1
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

    Returns
    -------
    out : Tensor [shape=(..., T)]
        The reconstructed waveform.

    """
    return nn.GriffinLim._func(
        y,
        out_length=out_length,
        frame_length=frame_length,
        frame_period=frame_period,
        fft_length=fft_length,
        center=center,
        mode=mode,
        window=window,
        norm=norm,
        symmetric=symmetric,
        n_iter=n_iter,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        init_phase=init_phase,
        verbose=verbose,
    )


def grpdelay(
    b: Tensor | None = None,
    a: Tensor | None = None,
    *,
    fft_length: int = 512,
    alpha: float = 1,
    gamma: float = 1,
    **kwargs,
) -> Tensor:
    """Compute group delay.

    Parameters
    ----------
    b : Tensor [shape=(..., M+1)] or None
        The numerator coefficients.

    a : Tensor [shape=(..., N+1)] or None
        The denominator coefficients.

    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    alpha : float > 0
        The tuning parameter, :math:`\\alpha`.

    gamma : float > 0
        The tuning parameter, :math:`\\gamma`.

    Returns
    -------
    out : Tensor [shape=(..., L/2+1)]
        The group delay or modified group delay function.

    """
    return nn.GroupDelay._func(
        b, a, fft_length=fft_length, alpha=alpha, gamma=gamma, **kwargs
    )


def hilbert(x: Tensor, fft_length: int | None = None, dim: int = -1) -> Tensor:
    """Compute the analytic signal using the Hilbert transform.

    Parameters
    ----------
    x : Tensor [shape=(..., T, ...)]
        The input signal.

    fft_length : int >= 1 or None
        The number of FFT bins. If None, set to :math:`T`.

    dim : int
        The dimension along which to take the Hilbert transform.

    Returns
    -------
    out : Tensor [shape=(..., T, ...)]
        The analytic signal.

    """
    return nn.HilbertTransform._func(x, fft_length=fft_length, dim=dim)


def hilbert2(
    x: Tensor,
    fft_length: ArrayLike[int] | int | None = None,
    dim: ArrayLike[int] = (-2, -1),
) -> Tensor:
    """Compute the analytic signal using the Hilbert transform.

    Parameters
    ----------
    x : Tensor [shape=(..., T1, T2, ...)]
        The input signal.

    fft_length : int, list[int], or None
        The number of FFT bins. If None, set to (:math:`T1`, :math:`T2`).

    dim : list[int]
        The dimensions along which to take the Hilbert transform.

    Returns
    -------
    out : Tensor [shape=(..., T1, T2, ...)]
        The analytic signal.

    """
    return nn.TwoDimensionalHilbertTransform._func(x, fft_length=fft_length, dim=dim)


def histogram(
    x: Tensor,
    n_bin: int = 10,
    lower_bound: float = 0,
    upper_bound: float = 1,
    norm: bool = False,
    softness: float = 1e-3,
) -> Tensor:
    """Compute histogram.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        The input data.

    n_bin : int >= 1
        The number of bins, :math:`K`.

    lower_bound : float < U
        The lower bound of the histogram, :math:`L`.

    upper_bound : float > L
        The upper bound of the histogram, :math:`U`.

    norm : bool
        If True, normalizes the histogram.

    softness : float > 0
        A smoothing parameter. The smaller value makes the output closer to the true
        histogram, but the gradient vanishes.

    Returns
    -------
    out : Tensor [shape=(..., K)]
        The histogram.

    """
    return nn.Histogram._func(
        x,
        n_bin=n_bin,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        norm=norm,
        softness=softness,
    )


def ialaw(y: Tensor, abs_max: float = 1, a: float = 87.6) -> Tensor:
    """Expand the waveform using the A-law algorithm.

    Parameters
    ----------
    y : Tensor [shape=(...,)]
        The compressed waveform.

    abs_max : float > 0
        The absolute maximum value of the original input waveform.

    a : float >= 1
        The compression factor, :math:`A`.

    Returns
    -------
    out : Tensor [shape=(...,)]
        The expanded waveform.

    """
    return nn.ALawExpansion._func(y, abs_max=abs_max, a=a)


def idct(y: Tensor, dct_type: int = 2) -> Tensor:
    """Compute inverse DCT.

    Parameters
    ----------
    y : Tensor [shape=(..., L)]
        The input.

    dct_type : int in [1, 4]
        The DCT type.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        The inverse DCT output.

    """
    return nn.InverseDiscreteCosineTransform._func(y, dct_type=dct_type)


def idht(y: Tensor, dht_type: int = 2) -> Tensor:
    """Compute inverse DHT.

    Parameters
    ----------
    y : Tensor [shape=(..., L)]
        The input.

    dht_type : int in [1, 4]
        The DHT type.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        The inverse DHT output.

    """
    return nn.InverseDiscreteHartleyTransform._func(y, dht_type=dht_type)


def idst(y: Tensor, dst_type: int = 2) -> Tensor:
    """Compute inverse DST.

    Parameters
    ----------
    y : Tensor [shape=(..., L)]
        The input.

    dst_type : int in [1, 4]
        The DST type.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        The inverse DST output.

    """
    return nn.InverseDiscreteSineTransform._func(y, dst_type=dst_type)


def ifbank(
    y: Tensor,
    fft_length: int,
    sample_rate: int,
    f_min: float = 0,
    f_max: float | None = None,
    gamma: float = 0,
    scale: str = "htk",
    erb_factor: float | None = None,
    use_power: bool = False,
) -> Tensor:
    """Reconstruct the power spectrum from the mel filter bank output.

    Parameters
    ----------
    y : Tensor [shape=(..., C)]
        The mel filter bank output.

    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    sample_rate : int >= 1
        The sample rate in Hz.

    f_min : float >= 0
        The minimum frequency in Hz.

    f_max : float <= sample_rate // 2
        The maximum frequency in Hz.

    gamma : float in [-1, 1]
        The parameter of the generalized logarithmic function.

    scale : ['htk', 'mel', 'inverted-mel', 'bark', 'linear']
        The type of auditory scale used to construct the filter bank.

    erb_factor : float > 0 or None
        The scale factor for the ERB scale, referred to as the E-factor. If not None,
        the filter bandwidths are adjusted according to the scaled ERB scale.

    use_power : bool
        Set to True if the mel filter bank output is extracted from the power spectrum
        instead of the amplitude spectrum.

    Returns
    -------
    out : Tensor [shape=(..., L/2+1)]
        The reconstructed power spectrum.

    """
    return nn.InverseMelFilterBankAnalysis._func(
        y,
        fft_length=fft_length,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        gamma=gamma,
        scale=scale,
        erb_factor=erb_factor,
        use_power=use_power,
    )


def ifftr(y: Tensor, out_length: int | None = None) -> Tensor:
    """Compute inverse FFT of a complex spectrum.

    Parameters
    ----------
    y : Tensor [shape=(..., L/2+1)]
        The complex input spectrum.

    out_length : int or None
        The output length, :math:`N`.

    Returns
    -------
    out : Tensor [shape=(..., N)]
        The real output signal.

    """
    return nn.RealValuedInverseFastFourierTransform._func(y, out_length=out_length)


def ifreqt2(
    c: Tensor, out_order: int, alpha: float = 0, theta: float = 0, n_fft: int = 512
) -> Tensor:
    """Perform second-order all-pass inverse frequency transform.

    Parameters
    ----------
    c : Tensor [shape=(..., M1+1)]
        The cepstral coefficients.

    out_order : int >= 0
        The order of the output sequence, :math:`M_2`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    theta : float in [0, 1]
        The emphasis frequency, :math:`\\theta`.

    n_fft : int >> M2
        The number of FFT bins.

    Returns
    -------
    out : Tensor [shape=(..., M2+1)]
        The warped cepstral coefficients.

    """
    return nn.SecondOrderAllPassInverseFrequencyTransform._func(
        c, out_order=out_order, alpha=alpha, theta=theta, n_fft=n_fft
    )


def ignorm(y: Tensor, gamma: float = 0, c: int | None = None) -> Tensor:
    """Perform cepstrum inverse gain normalization.

    Parameters
    ----------
    y : Tensor [shape=(..., M+1)]
        The normalized generalized cepstrum.

    gamma : float in [-1, 1]
        The gamma parameter, :math:`\\gamma`.

    c : int >= 1 or None
        The number of filter stages.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The generalized cepstrum.

    """
    return nn.GeneralizedCepstrumInverseGainNormalization._func(y, gamma=gamma, c=c)


def imdct(
    y: Tensor,
    out_length: int | None = None,
    frame_length: int = 400,
    window: str = "sine",
) -> Tensor:
    """Compute inverse modified discrete cosine transform.

    Parameters
    ----------
    y : Tensor [shape=(..., 2T/L, L/2)]
        The spectrum.

    out_length : int or None
        The length of the output waveform.

    frame_length : int >= 2
        The frame length, :math:`L`.

    window : ['sine', 'vorbis', 'kbd', 'rectangular']
        The window type.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        The reconstructed waveform.

    """
    return nn.InverseModifiedDiscreteCosineTransform._func(
        y, out_length=out_length, frame_length=frame_length, window=window
    )


def imdst(
    y: Tensor,
    out_length: int | None = None,
    frame_length: int = 400,
    window: str = "sine",
) -> Tensor:
    """Compute inverse modified discrete sine transform.

    Parameters
    ----------
    y : Tensor [shape=(..., 2T/L, L/2)]
        The spectrum.

    out_length : int or None
        The length of the output waveform.

    frame_length : int >= 2
        The frame length, :math:`L`.

    window : ['sine', 'vorbis', 'kbd', 'rectangular']
        The window type.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        The reconstructed waveform.

    """
    return nn.InverseModifiedDiscreteSineTransform._func(
        y, out_length=out_length, frame_length=frame_length, window=window
    )


def interpolate(x: Tensor, period: int = 1, start: int = 0, dim: int = -1) -> Tensor:
    """Interpolate the input signal.

    Parameters
    ----------
    x : Tensor [shape=(..., T, ...)]
        The input signal.

    period : int >= 1
        The interpolation period, :math:`P`.

    start : int >= 0
        The start point, :math:`S`.

    dim : int
        The dimension along which to interpolate the tensors.

    Returns
    -------
    out : Tensor [shape=(..., TxP+S, ...)]
        The interpolated signal.

    """
    return nn.Interpolation._func(x, period=period, start=start, dim=dim)


def ipnorm(y: Tensor) -> Tensor:
    """Perform cepstrum inverse power normalization.

    Parameters
    ----------
    y : Tensor [shape=(..., M+2)]
        The log power and power-normalized cepstrum.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The cepstrum.

    """
    return nn.MelCepstrumInversePowerNormalization._func(y)


def is2par(s: Tensor) -> Tensor:
    """Convert IS to PARCOR.

    Parameters
    ----------
    s : Tensor [shape=(..., M+1)]
        The inverse sine coefficients.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The PARCOR coefficients.

    """
    return nn.InverseSineToParcorCoefficients._func(s)


def istft(
    y: Tensor,
    *,
    out_length: int | None = None,
    frame_length: int = 400,
    frame_period: int = 80,
    fft_length: int = 512,
    center: bool = True,
    window: str = "blackman",
    norm: str = "power",
    symmetric: bool = True,
) -> Tensor:
    """Compute inverse short-time Fourier transform.

    Parameters
    ----------
    y : Tensor [shape=(..., T/P, N/2+1)]
        The complex spectrogram.

    out_length : int >= 1 or None
        The length of the output waveform.

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

    Returns
    -------
    out : Tensor [shape=(..., T)]
        The reconstructed waveform.

    """
    return nn.InverseShortTimeFourierTransform._func(
        y,
        out_length=out_length,
        frame_length=frame_length,
        frame_period=frame_period,
        fft_length=fft_length,
        center=center,
        window=window,
        norm=norm,
        symmetric=symmetric,
    )


def iulaw(y: Tensor, abs_max: float = 1, mu: int = 255) -> Tensor:
    """Expand the waveform using the :math:`\\mu`-law algorithm.

    Parameters
    ----------
    y : Tensor [shape=(...,)]
        The compressed waveform.

    abs_max : float > 0
        The absolute maximum value of the original input waveform.

    mu : int >= 1
        The compression factor, :math:`\\mu`.

    Returns
    -------
    out : Tensor [shape=(...,)]
        The expanded waveform.

    """
    return nn.MuLawExpansion._func(y, abs_max=abs_max, mu=mu)


def iwht(y: Tensor, wht_type: str = "natural") -> Tensor:
    """Compute inverse WHT.

    Parameters
    ----------
    y : Tensor [shape=(..., L)]
        The input.

    wht_type : ['sequency', 'natural', 'dyadic']
        The order of the coefficients in the Walsh matrix.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        The inverse WHT output.

    """
    return nn.InverseWalshHadamardTransform._func(y, wht_type=wht_type)


def lar2par(g: Tensor) -> Tensor:
    """Convert LAR to PARCOR.

    Parameters
    ----------
    g : Tensor [shape=(..., M+1)]
        The log area ratio.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The PARCOR coefficients.

    """
    return nn.LogAreaRatioToParcorCoefficients._func(g)


def levdur(r: Tensor, eps: float = 0) -> Tensor:
    """Solve a Yule-Walker linear system.

    Parameters
    ----------
    r : Tensor [shape=(..., M+1)]
        The autocorrelation.

    eps : float >= 0
        A small value to improve numerical stability.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The gain and LPC coefficients.

    """
    return nn.LevinsonDurbin._func(r, eps=eps)


def linear_intpl(x: Tensor, upsampling_factor: int = 80) -> Tensor:
    """Interpolate filter coefficients.

    Parameters
    ----------
    x : Tensor [shape=(B, N, D) or (N, D) or (N,)]
        The filter coefficients.

    upsampling_factor : int >= 1
        The upsampling factor, :math:`P`.

    Returns
    -------
    y : Tensor [shape=(B, NxP, D) or (NxP, D) or (NxP,)]
        The upsampled filter coefficients.

    """
    return nn.LinearInterpolation._func(x, upsampling_factor=upsampling_factor)


def lpc(x: Tensor, lpc_order: int, eps: float = 1e-6) -> Tensor:
    """Perform LPC analysis.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        The famed waveform.

    lpc_order : int >= 0
        The order of the LPC coefficients, :math:`M`.

    eps : float >= 0
        A small value to improve numerical stability.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The gain and LPC coefficients.

    """
    return nn.LinearPredictiveCodingAnalysis._func(x, lpc_order=lpc_order, eps=eps)


def lpc2lsp(
    a: Tensor,
    log_gain: bool = False,
    sample_rate: int | None = None,
    out_format: str = "radian",
) -> Tensor:
    """Convert LPC to LSP.

    Parameters
    ----------
    a : Tensor [shape=(..., M+1)]
        The LPC coefficients.

    log_gain : bool
        If True, output the gain in logarithmic scale.

    sample_rate : int >= 1 or None
        The sample rate in Hz.

    out_format : ['radian', 'cycle', 'khz', 'hz']
        The output format.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The LSP frequencies.

    """
    return nn.LinearPredictiveCoefficientsToLineSpectralPairs._func(
        a, log_gain=log_gain, sample_rate=sample_rate, out_format=out_format
    )


def lpc2par(a: Tensor, gamma: float = 1, c: int | None = None) -> Tensor:
    """Convert LPC to PARCOR.

    Parameters
    ----------
    a : Tensor [shape=(..., M+1)]
        The LPC coefficients.

    gamma : float in [-1, 1]
        The gamma parameter, :math:`\\gamma`.

    c : int >= 1 or None
        The number of filter stages.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The PARCOR coefficients.

    """
    return nn.LinearPredictiveCoefficientsToParcorCoefficients._func(
        a, gamma=gamma, c=c
    )


def lpccheck(a: Tensor, margin: float = 1e-16, warn_type: str = "warn") -> Tensor:
    """Check stability of LPC coefficients.

    Parameters
    ----------
    a : Tensor [shape=(..., M+1)]
        The input LPC coefficients.

    margin : float in (0, 1)
        The margin to guarantee the stability of LPC.

    warn_type : ['ignore', 'warn', 'exit']
        The warning type.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The modified LPC coefficients.

    """
    return nn.LinearPredictiveCoefficientsStabilityCheck._func(
        a, margin=margin, warn_type=warn_type
    )


def lsp2lpc(
    w: Tensor,
    log_gain: bool = False,
    sample_rate: int | None = None,
    in_format: str = "radian",
) -> Tensor:
    """Convert LSP to LPC.

    Parameters
    ----------
    w : Tensor [shape=(..., M+1)]
        The LSP frequencies.

    log_gain : bool
        If True, assume the input gain is in logarithmic scale.

    sample_rate : int >= 1 or None
        The sample rate in Hz.

    in_format : ['radian', 'cycle', 'khz', 'hz']
        The input format.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The LPC coefficients.

    """
    return nn.LineSpectralPairsToLinearPredictiveCoefficients._func(
        w, log_gain=log_gain, sample_rate=sample_rate, in_format=in_format
    )


def lspcheck(
    w: Tensor, rate: float = 0, n_iter: int = 1, warn_type: str = "warn"
) -> Tensor:
    """Check the stability of the input LSP coefficients.

    Parameters
    ----------
    w : Tensor [shape=(..., M+1)]
        The input LSP coefficients in radians.

    rate : float in [0, 1]
        The rate of distance between two adjacent LSPs.

    n_iter : int >= 0
        The number of iterations for the modification.

    warn_type : ['ignore', 'warn', 'exit']
        The warning type.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The modified LSP frequencies.

    """
    return nn.LineSpectralPairsStabilityCheck._func(
        w, rate=rate, n_iter=n_iter, warn_type=warn_type
    )


def lsp2sp(
    w: Tensor,
    fft_length: int,
    alpha: float = 0,
    gamma: float = -1,
    log_gain: bool = False,
    out_format: str = "power",
) -> Tensor:
    """Convert line spectral pairs to spectrum.

    Parameters
    ----------
    w : Tensor [shape=(..., M+1)]
        The line spectral pairs in radians.

    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    alpha : float in (-1, 1)
        The warping factor, :math:`\\alpha`.

    gamma : float in [-1, 0)
        The gamma parameter, :math:`\\gamma`.

    log_gain : bool
        If True, assume the input gain is in logarithmic scale.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        The output format.

    Returns
    -------
    out : Tensor [shape=(..., L/2+1)]
        The spectrum.

    """
    return nn.LineSpectralPairsToSpectrum._func(
        w,
        fft_length=fft_length,
        alpha=alpha,
        gamma=gamma,
        log_gain=log_gain,
        out_format=out_format,
    )


def magic_intpl(x: Tensor, magic_number: float = 0) -> Tensor:
    """Interpolate magic number.

    Parameters
    ----------
    x : Tensor [shape=(B, N, D) or (N, D) or (N,)]
        The data containing magic number.

    magic_number : float or Tensor
        The magic number to be interpolated.

    Returns
    -------
    out : Tensor [shape=(B, N, D) or (N, D) or (N,)]
        The data after interpolation.

    """
    return nn.MagicNumberInterpolation._func(x, magic_number=magic_number)


def mc2b(mc: Tensor, alpha: float = 0) -> Tensor:
    """Convert mel-cepstrum to MLSA digital filter coefficients.

    Parameters
    ----------
    mc : Tensor [shape=(..., M+1)]
        The mel-cepstral coefficients.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The MLSA digital filter coefficients.

    """
    return nn.MelCepstrumToMLSADigitalFilterCoefficients._func(mc, alpha=alpha)


def mcep(x: Tensor, cep_order: int, alpha: float = 0, n_iter: int = 0) -> Tensor:
    """Perform mel-cepstral analysis.

    Parameters
    ----------
    x : Tensor [shape=(..., L/2+1)]
        The power spectrum.

    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    n_iter : int >= 0
        The number of iterations.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The mel-cepstrum.

    """
    return nn.MelCepstralAnalysis._func(
        x, cep_order=cep_order, alpha=alpha, n_iter=n_iter
    )


def mcpf(
    mc: Tensor, alpha: float = 0, beta: float = 0, onset: int = 2, ir_length: int = 128
) -> Tensor:
    """Perform mel-cesptrum postfiltering.

    Parameters
    ----------
    mc : Tensor [shape=(..., M+1)]
        The input mel-cepstral coefficients.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    beta : float
        The intensity parameter, :math:`\\beta`.

    onset : int >= 0
        The onset index.

    ir_length : int >= 1
        The length of the impulse response.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The postfiltered mel-cepstral coefficients.

    """
    return nn.MelCepstrumPostfiltering._func(
        mc, alpha=alpha, beta=beta, onset=onset, ir_length=ir_length
    )


def mdct(x: Tensor, frame_length: int = 400, window: str = "sine") -> Tensor:
    """Compute modified discrete cosine transform.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        The input waveform.

    frame_length : int >= 2
        The frame length, :math:`L`.

    window : ['sine', 'vorbis', 'kbd', 'rectangular']
        The window type.

    Returns
    -------
    out : Tensor [shape=(..., 2T/L, L/2)]
        The spectrum.

    """
    return nn.ModifiedDiscreteCosineTransform._func(
        x, frame_length=frame_length, window=window
    )


def mdst(x: Tensor, frame_length: int = 400, window: str = "sine") -> Tensor:
    """Compute modified discrete sine transform.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        The input waveform.

    frame_length : int >= 2
        The frame length, :math:`L`.

    window : ['sine', 'vorbis', 'kbd', 'rectangular']
        The window type.

    Returns
    -------
    out : Tensor [shape=(..., 2T/L, L/2)]
        The spectrum.

    """
    return nn.ModifiedDiscreteSineTransform._func(
        x, frame_length=frame_length, window=window
    )


def mfcc(
    x: Tensor,
    mfcc_order: int,
    n_channel: int,
    sample_rate: int,
    lifter: int = 1,
    f_min: float = 0,
    f_max: float | None = None,
    floor: float = 1e-5,
    gamma: float = 0,
    scale: str = "htk",
    erb_factor: float | None = None,
    out_format: str = "y",
) -> Tensor:
    """Compute the MFCC from the power spectrum.

    Parameters
    ----------
    x : Tensor [shape=(..., L/2+1)]
        The power spectrum.

    mfcc_order : int >= 1
        The order of the MFCC, :math:`M`.

    n_channel : int >= 1
        The number of mel filter banks, :math:`C`.

    sample_rate : int >= 1
        The sample rate in Hz.

    lifter : int >= 1
        The liftering coefficient.

    f_min : float >= 0
        The minimum frequency in Hz.

    f_max : float <= sample_rate // 2
        The maximum frequency in Hz.

    floor : float > 0
        The minimum mel filter bank output in linear scale.

    gamma : float in [-1, 1]
        The parameter of the generalized logarithmic function.

    scale : ['htk', 'mel', 'inverted-mel', 'bark', 'linear']
        The type of auditory scale used to construct the filter bank.

    erb_factor : float > 0 or None
        The scale factor for the ERB scale, referred to as the E-factor. If not None,
        the filter bandwidths are adjusted according to the scaled ERB scale.

    out_format : ['y', 'yE', 'yc', 'ycE']
        `y` is MFCC, `c` is C0, and `E` is energy.

    Returns
    -------
    y : Tensor [shape=(..., M)]
        The MFCC without C0.

    E : Tensor [shape=(..., 1)] (optional)
        The energy.

    c : Tensor [shape=(..., 1)] (optional)
        The C0.

    """
    return nn.MelFrequencyCepstralCoefficientsAnalysis._func(
        x,
        mfcc_order=mfcc_order,
        n_channel=n_channel,
        sample_rate=sample_rate,
        lifter=lifter,
        f_min=f_min,
        f_max=f_max,
        floor=floor,
        gamma=gamma,
        scale=scale,
        erb_factor=erb_factor,
        out_format=out_format,
    )


def mgc2mgc(
    mc: Tensor,
    out_order: int,
    in_alpha: float = 0,
    out_alpha: float = 0,
    in_gamma: float = 0,
    out_gamma: float = 0,
    in_norm: bool = False,
    out_norm: bool = False,
    in_mul: bool = False,
    out_mul: bool = False,
    n_fft: int = 512,
) -> Tensor:
    """Convert mel-generalized cepstrum to mel-generalized cepstrum.

    Parameters
    ----------
    mc : Tensor [shape=(..., M1+1)]
        The input mel-cepstrum.

    out_order : int >= 0
        The order of the output cepstrum, :math:`M_2`.

    in_alpha : float in (-1, 1)
        The input alpha, :math:`\\alpha_1`.

    out_alpha : float in (-1, 1)
        The output alpha, :math:`\\alpha_2`.

    in_gamma : float in [-1, 1]
        The input gamma, :math:`\\gamma_1`.

    out_gamma : float in [-1, 1]
        The output gamma, :math:`\\gamma_2`.

    in_norm : bool
        If True, the input is assumed to be normalized.

    out_norm : bool
        If True, the output is assumed to be normalized.

    in_mul : bool
        If True, the input is assumed to be gamma-multiplied.

    out_mul : bool
        If True, the output is assumed to be gamma-multiplied.

    n_fft : int >> M1, M2
        The number of FFT bins.

    Returns
    -------
    out : Tensor [shape=(..., M2+1)]
        The converted mel-cepstrum.

    """
    return nn.MelGeneralizedCepstrumToMelGeneralizedCepstrum._func(
        mc,
        out_order=out_order,
        in_alpha=in_alpha,
        out_alpha=out_alpha,
        in_gamma=in_gamma,
        out_gamma=out_gamma,
        in_norm=in_norm,
        out_norm=out_norm,
        in_mul=in_mul,
        out_mul=out_mul,
        n_fft=n_fft,
    )


def mgc2sp(
    mc: Tensor,
    fft_length: int,
    alpha: float = 0,
    gamma: float = 0,
    norm: bool = False,
    mul: bool = False,
    n_fft: int = 512,
    out_format: str = "power",
) -> Tensor:
    """Convert mel-cepstrum to spectrum.

    Parameters
    ----------
    mc : Tensor [shape=(..., M+1)]
        Mel-cepstrum.

    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    gamma : float in [-1, 1]
        The gamma parameter, :math:`\\gamma`.

    norm : bool
        If True, the input is assumed to be normalized.

    mul : bool
        If True, the input is assumed to be gamma-multiplied.

    n_fft : int >> L
        The number of FFT bins.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power', \
                  'cycle', 'radian', 'degree', 'complex']
        The output format.

    Returns
    -------
    out : Tensor [shape=(..., L/2+1)]
        Spectrum.

    """
    return nn.MelGeneralizedCepstrumToSpectrum._func(
        mc,
        fft_length=fft_length,
        alpha=alpha,
        gamma=gamma,
        norm=norm,
        mul=mul,
        n_fft=n_fft,
        out_format=out_format,
    )


def mlpg(
    u: Tensor,
    seed: ArrayLike[ArrayLike[float]] | ArrayLike[int] = [[-0.5, 0, 0.5], [1, -2, 1]],
) -> Tensor:
    """Perform MLPG given the mean vectors with delta components.

    Parameters
    ----------
    u : Tensor [shape=(..., T, DxH)]
        The time-variant mean vectors with delta components.

    seed : list[list[float]] or list[int]
        The delta coefficients or the width(s) of 1st (and 2nd) regression coefficients.

    Returns
    -------
    out : Tensor [shape=(..., T, D)]
        The smoothed static components.

    """
    return nn.MaximumLikelihoodParameterGeneration._func(u, seed=seed)


def mlsacheck(
    c: Tensor,
    *,
    alpha: float = 0,
    pade_order: int = 4,
    strict: bool = True,
    threshold: float | None = None,
    fast: bool = True,
    n_fft: int = 256,
    warn_type: str = "warn",
    mod_type: str = "scale",
) -> Tensor:
    """Check the stability of the MLSA digital filter.

    Parameters
    ----------
    c : Tensor [shape=(..., M+1)]
        The input Mel-cepstrum.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    pade_order : int in [4, 7]
        The order of the Pade approximation.

    strict : bool
        If True, prioritizes maintaining the maximum log approximation error over MLSA
        filter stability.

    threshold : float > 0 or None
        The threshold value. If None, it is automatically computed.

    fast : bool
        Enables fast mode (do not use FFT).

    n_fft : int > M
        The number of FFT bins. Used only in non-fast mode.

    warn_type : ['ignore', 'warn', 'exit']
        The warning type.

    mod_type : ['clip', 'scale']
        The modification method.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The modified mel-cepstrum.

    """
    return nn.MLSADigitalFilterStabilityCheck._func(
        c,
        alpha=alpha,
        pade_order=pade_order,
        strict=strict,
        threshold=threshold,
        fast=fast,
        n_fft=n_fft,
        warn_type=warn_type,
        mod_type=mod_type,
    )


def mpir2c(h: Tensor, cep_order: int, n_fft: int = 512) -> Tensor:
    """Convert minimum-phase impulse response to cepstrum.

    Parameters
    ----------
    h : Tensor [shape=(..., N)]
        The truncated minimum-phase impulse response.

    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    n_fft : int >> N
        The number of FFT bins used for conversion. The accurate conversion requires the
        large value.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The cepstral coefficients.

    """
    return nn.MinimumPhaseImpulseResponseToCepstrum._func(
        h, cep_order=cep_order, n_fft=n_fft
    )


def ndps2c(n: Tensor, cep_order: int) -> Tensor:
    """Convert NPDS to cepstrum.

    Parameters
    ----------
    n : Tensor [shape=(..., L/2+1)]
        The NDPS, where :math:`L` is the number of FFT bins.

    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The cepstrum.

    """
    return nn.NegativeDerivativeOfPhaseSpectrumToCepstrum._func(n, cep_order=cep_order)


def norm0(a: Tensor) -> Tensor:
    """Convert all-pole to all-zero filter coefficients vice versa.

    Parameters
    ----------
    a : Tensor [shape=(..., M+1)]
        The all-pole or all-zero filter coefficients.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The all-zero or all-pole filter coefficients.

    """
    return nn.AllPoleToAllZeroDigitalFilterCoefficients._func(a)


def par2is(k: Tensor) -> Tensor:
    """Convert PARCOR to IS.

    Parameters
    ----------
    k : Tensor [shape=(..., M+1)]
        The PARCOR coefficients.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The inverse sine coefficients.

    """
    return nn.ParcorCoefficientsToInverseSine._func(k)


def par2lar(k: Tensor) -> Tensor:
    """Convert PARCOR to LAR.

    Parameters
    ----------
    k : Tensor [shape=(..., M+1)]
        The PARCOR coefficients.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The log area ratio.

    """
    return nn.ParcorCoefficientsToLogAreaRatio._func(k)


def par2lpc(k: Tensor, gamma: float = 1, c: int | None = None) -> Tensor:
    """Convert PARCOR to LPC.

    Parameters
    ----------
    k : Tensor [shape=(..., M+1)]
        The PARCOR coefficients.

    gamma : float in [-1, 1]
        The gamma parameter, :math:`\\gamma`.

    c : int >= 1 or None
        The number of filter stages.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The LPC coefficients.

    """
    return nn.ParcorCoefficientsToLinearPredictiveCoefficients._func(
        k, gamma=gamma, c=c
    )


def phase(
    b: Tensor | None = None,
    a: Tensor | None = None,
    *,
    fft_length: int = 512,
    unwrap: bool = False,
) -> Tensor:
    """Compute phase spectrum.

    Parameters
    ----------
    b : Tensor [shape=(..., M+1)] or None
        The numerator coefficients.

    a : Tensor [shape=(..., N+1)] or None
        The denominator coefficients.

    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    unwrap : bool
        If True, perform the phase unwrapping.

    Returns
    -------
    out : Tensor [shape=(..., L/2+1)]
        The phase spectrum [:math:`\\pi` rad].

    """
    return nn.Phase._func(b, a, fft_length=fft_length, unwrap=unwrap)


def plp(
    x: Tensor,
    plp_order: int,
    n_channel: int,
    sample_rate: int,
    compression_factor: float = 0.33,
    lifter: int = 1,
    f_min: float = 0,
    f_max: float | None = None,
    floor: float = 1e-5,
    gamma: float = 0,
    scale: str = "htk",
    erb_factor: float | None = None,
    n_fft: int = 512,
    out_format: str = "y",
) -> Tensor:
    """Compute the MFCC from the power spectrum.

    Parameters
    ----------
    x : Tensor [shape=(..., L/2+1)]
        The power spectrum.

    plp_order : int >= 1
        The order of the PLP, :math:`M`.

    n_channel : int >= 1
        The number of mel filter banks, :math:`C`.

    sample_rate : int >= 1
        The sample rate in Hz.

    compression_factor : float > 0
        The amplitude compression factor.

    lifter : int >= 1
        The liftering coefficient.

    f_min : float >= 0
        The minimum frequency in Hz.

    f_max : float <= sample_rate // 2
        The maximum frequency in Hz.

    floor : float > 0
        The minimum mel filter bank output in linear scale.

    gamma : float in [-1, 1]
        The parameter of the generalized logarithmic function.

    scale : ['htk', 'mel', 'inverted-mel', 'bark', 'linear']
        The type of auditory scale used to construct the filter bank.

    erb_factor : float > 0 or None
        The scale factor for the ERB scale, referred to as the E-factor. If not None,
        the filter bandwidths are adjusted according to the scaled ERB scale.

    n_fft : int >> M
        The number of FFT bins for the conversion from LPC to cepstrum.

    out_format : ['y', 'yE', 'yc', 'ycE']
        `y` is MFCC, `c` is C0, and `E` is energy.

    Returns
    -------
    y : Tensor [shape=(..., M)]
        The MFCC without C0.

    E : Tensor [shape=(..., 1)] (optional)
        The energy.

    c : Tensor [shape=(..., 1)] (optional)
        The C0.

    """
    return nn.PerceptualLinearPredictiveCoefficientsAnalysis._func(
        x,
        plp_order=plp_order,
        n_channel=n_channel,
        sample_rate=sample_rate,
        compression_factor=compression_factor,
        lifter=lifter,
        f_min=f_min,
        f_max=f_max,
        floor=floor,
        gamma=gamma,
        scale=scale,
        erb_factor=erb_factor,
        n_fft=n_fft,
        out_format=out_format,
    )


def pnorm(x: Tensor, alpha: float = 0, ir_length: int = 128) -> Tensor:
    """Perform cepstrum power normalization.

    Parameters
    ----------
    x : Tensor [shape=(..., M+1)]
        The input mel-cepstrum.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    ir_length : int >= 1
        The length of the impulse response.

    Returns
    -------
    out : Tensor [shape=(..., M+2)]
        The power-normalized cepstrum.

    """
    return nn.MelCepstrumPowerNormalization._func(x, alpha=alpha, ir_length=ir_length)


def pol_root(
    x: Tensor, *, eps: float | None = None, in_format: str = "rectangular"
) -> Tensor:
    """Convert roots to polynomial coefficients.

    Parameters
    ----------
    x : Tensor [shape=(..., M)]
        The roots, can be complex.

    eps : float >= 0 or None
        If the absolute values of the imaginary parts of the polynomial coefficients are
        all less than this value, they are considered as real numbers.

    in_format : ['rectangular', 'polar']
        The input format.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The polynomial coefficients.

    """
    return nn.RootsToPolynomial._func(x, eps=eps, in_format=in_format)


def poledf(
    x: Tensor, a: Tensor, frame_period: int = 80, ignore_gain: bool = False
) -> Tensor:
    """Apply an all-pole digital filter.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        The excitation signal.

    a : Tensor [shape=(..., T/P, M+1)]
        The filter coefficients.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    ignore_gain : bool
        If True, perform filtering without the gain.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        The output signal.

    """
    return nn.AllPoleDigitalFilter._func(
        x, a, frame_period=frame_period, ignore_gain=ignore_gain
    )


def quantize(
    x: Tensor, abs_max: float = 1, n_bit: int = 8, quantizer: str = "mid-rise"
) -> Tensor:
    """Quantize the input waveform.

    Parameters
    ----------
    x : Tensor [shape=(...,)]
        The input waveform.

    abs_max : float > 0
        The absolute maximum value of the input waveform.

    n_bit : int >= 1
        The number of quantization bits.

    quantizer : ['mid-rise', 'mid-tread']
        The quantizer type.

    Returns
    -------
    out : Tensor [shape=(...,)]
        The quantized waveform.

    """
    return nn.UniformQuantization._func(
        x, abs_max=abs_max, n_bit=n_bit, quantizer=quantizer
    )


def rlevdur(a: Tensor) -> Tensor:
    """Solve a Yule-Walker linear system given the LPC coefficients.

    Parameters
    ----------
    a : Tensor [shape=(..., M+1)]
        The gain and the LPC coefficients.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The autocorrelation.

    """
    return nn.ReverseLevinsonDurbin._func(a)


def rmse(x: Tensor, y: Tensor, reduction: str = "mean") -> Tensor:
    """Calculate RMSE.

    Parameters
    ----------
    x : Tensor [shape=(..., D)]
        The input.

    y : Tensor [shape=(..., D)]
        The target.

    reduction : ['none', 'mean', 'sum']
        The reduction type.

    Returns
    -------
    out : Tensor [shape=(...,) or scalar]
        The RMSE.

    """
    return nn.RootMeanSquareError._func(x, y, reduction=reduction)


def root_pol(
    a: Tensor, *, eps: float | None = None, out_format: str = "rectangular"
) -> Tensor:
    """Compute roots of polynomial.

    Parameters
    ----------
    a : Tensor [shape=(..., M+1)]
        The polynomial coefficients.

    eps : float >= 0 or None
        If the absolute values of the imaginary parts of the roots are all less than
        this value, they are considered as real roots.

    out_format : ['rectangular', 'polar']
        Output format.

    Returns
    -------
    out : Tensor [shape=(..., M)]
        The roots.

    """
    return nn.PolynomialToRoots._func(
        a,
        eps=eps,
        out_format=out_format,
    )


def smcep(
    x: Tensor,
    cep_order: int,
    alpha: float = 0,
    theta: float = 0,
    n_iter: int = 0,
    accuracy_factor: int = 4,
) -> Tensor:
    """Perform mel-cepstral analysis.

    Parameters
    ----------
    x : Tensor [shape=(..., L/2+1)]
        The power spectrum.

    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    theta : float in [0, 1]
        The emphasis frequency, :math:`\\theta`.

    n_iter : int >= 0
        The number of iterations.

    accuracy_factor : int >= 1
        The accuracy factor multiplied by the FFT length.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        The mel-cepstrum.

    """
    return nn.SecondOrderAllPassMelCepstralAnalysis._func(
        x,
        cep_order=cep_order,
        alpha=alpha,
        theta=theta,
        n_iter=n_iter,
        accuracy_factor=accuracy_factor,
    )


def snr(
    s: Tensor,
    sn: Tensor,
    frame_length: int | None = None,
    full: bool = False,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> Tensor:
    """Calculate SNR.

    Parameters
    ----------
    s : Tensor [shape=(..., T)]
        The signal.

    sn : Tensor [shape=(..., T)]
        The signal with noise.

    frame_length : int >= 1 or None
        The frame length in samples, :math:`L`. If given, calculate the segmental SNR.

    full : bool
        If True, include the constant term in the SNR calculation.

    reduction : ['none', 'mean', 'sum']
        The reduction type.

    eps : float >= 0
        A small value to avoid NaN.

    Returns
    -------
    out : Tensor [shape=(...,) or scalar]
        The SNR.

    """
    return nn.SignalToNoiseRatio._func(
        s, sn, frame_length=frame_length, full=full, reduction=reduction, eps=eps
    )


def spec(
    b: Tensor | None = None,
    a: Tensor | None = None,
    *,
    fft_length: int = 512,
    eps: float = 0,
    relative_floor: float | None = None,
    out_format: str = "power",
) -> Tensor:
    """Compute spectrum.

    Parameters
    ----------
    b : Tensor [shape=(..., M+1)] or None
        The numerator coefficients.

    a : Tensor [shape=(..., N+1)] or None
        The denominator coefficients.

    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    eps : float >= 0
        A small value added to the power spectrum.

    relative_floor : float < 0 or None
        The relative floor of the power spectrum in dB.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        The output format.

    Returns
    -------
    out : Tensor [shape=(..., L/2+1)]
        The spectrum.

    """
    return nn.Spectrum._func(
        b,
        a,
        fft_length=fft_length,
        eps=eps,
        relative_floor=relative_floor,
        out_format=out_format,
    )


def stft(
    x: Tensor,
    *,
    frame_length: int = 400,
    frame_period: int = 80,
    fft_length: int = 512,
    center: bool = True,
    zmean: bool = False,
    mode: str = "constant",
    window: str = "blackman",
    norm: str = "power",
    symmetric: bool = True,
    eps: float = 1e-9,
    relative_floor: float | None = None,
    out_format: str = "power",
) -> Tensor:
    """Compute short-time Fourier transform.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        The input waveform.

    frame_length : int >= 1
        The frame length in samples, :math:`L`.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    fft_length : int >= L
        The number of FFT bins, :math:`N`.

    center : bool
        If True, pad the input on both sides so that the frame is centered.

    zmean : bool
        If True, perform mean subtraction on each frame.

    mode : ['constant', 'reflect', 'replicate', 'circular']
        The padding method.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular', 'nuttall']
        The window type.

    norm : ['none', 'power', 'magnitude']
        The normalization type of the window.

    symmetric : bool
        If True, the window is symmetric, otherwise periodic.

    eps : float >= 0
        A small value added to the power spectrum.

    relative_floor : float < 0 or None
        The relative floor of the power spectrum in dB.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power', 'complex']
        The output format.

    Returns
    -------
    out : Tensor [shape=(..., T/P, N/2+1)]
        The output spectrogram.

    """
    return nn.ShortTimeFourierTransform._func(
        x,
        frame_length=frame_length,
        frame_period=frame_period,
        fft_length=fft_length,
        center=center,
        zmean=zmean,
        mode=mode,
        window=window,
        norm=norm,
        symmetric=symmetric,
        eps=eps,
        relative_floor=relative_floor,
        out_format=out_format,
    )


def ulaw(x: Tensor, abs_max: float = 1, mu: int = 255) -> Tensor:
    """Compress the input waveform using the :math:`\\mu`-law algorithm.

    Parameters
    ----------
    x : Tensor [shape=(...,)]
        The input waveform.

    abs_max : float > 0
        The absolute maximum value of the input waveform.

    mu : int >= 1
        The compression factor, :math:`\\mu`.

    Returns
    -------
    out : Tensor [shape=(...,)]
        The compressed waveform.

    """
    return nn.MuLawCompression._func(x, abs_max=abs_max, mu=mu)


def unframe(
    y: Tensor,
    out_length: int | None = None,
    *,
    frame_period: int = 80,
    center: bool = True,
    window: str = "rectangular",
    norm: str = "none",
    symmetric: bool = True,
) -> Tensor:
    """Revert framed waveform.

    Parameters
    ----------
    y : Tensor [shape=(..., T/P, L)]
        The framed waveform.

    out_length : int >= 1 or None
        The length of the original waveform, :math:`T`.

    frame_peirod : int >= 1
        The frame period in samples, :math:`P`.

    center : bool
        If True, pad the input on both sides so that the frame is centered.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular', 'nuttall']
        The window type.

    norm : ['none', 'power', 'magnitude']
        The normalization type of the window.

    symmetric : bool
        If True, the window is symmetric, otherwise periodic.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        The unframed waveform.

    """
    return nn.Unframe._func(
        y,
        out_length=out_length,
        frame_period=frame_period,
        center=center,
        window=window,
        norm=norm,
        symmetric=symmetric,
    )


def wht(x: Tensor, wht_type: str = "natural") -> Tensor:
    """Apply WHT to the input.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        The input.

    wht_type : ['sequency', 'natural', 'dyadic']
        The order of the coefficients in the Walsh matrix.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        The WHT output.

    """
    return nn.WalshHadamardTransform._func(x, wht_type=wht_type)


def window(
    x: Tensor,
    out_length: int | None = None,
    *,
    window: str = "blackman",
    norm: str = "power",
    symmetric: bool = True,
) -> Tensor:
    """Apply a window function to the given waveform.

    Parameters
    ----------
    x : Tensor [shape=(..., L1)]
        The input framed waveform.

    out_length : int >= L1 or None
        The output length, :math:`L_2`. If :math:`L_2 > L_1`, output is zero-padded.
        If None, :math:`L_2 = L_1`.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular', 'nuttall', 'povey']
        The window type.

    norm : ['none', 'power', 'magnitude']
        The normalization type of the window.

    symmetric : bool
        If True, the window is symmetric, otherwise periodic.

    Returns
    -------
    out : Tensor [shape=(..., L2)]
        The windowed waveform.

    """
    return nn.Window._func(
        x, out_length=out_length, window=window, norm=norm, symmetric=symmetric
    )


def yingram(
    x: Tensor,
    sample_rate: int = 22050,
    lag_min: int = 22,
    lag_max: int | None = None,
    n_bin: int = 20,
) -> Tensor:
    """Compute the YIN derivatives from the waveform.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        The framed waveform.

    sample_rate : int >= 8000
        The sample rate in Hz.

    lag_min : int >= 1
        The minimum lag in points.

    lag_max : int < L
        The maximum lag in points.

    n_bin : int >= 1
        The number of bins to represent a semitone range.

    Returns
    -------
    out : Tensor [shape=(..., M)]
        The Yingram.

    """
    return nn.Yingram._func(
        x, sample_rate=sample_rate, lag_min=lag_min, lag_max=lag_max, n_bin=n_bin
    )


def zcross(
    x: Tensor, frame_length: int, norm: bool = False, softness: float = 1e-3
) -> Tensor:
    """Compute zero-crossing rate.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        The input waveform.

    frame_length : int >= 1
        The frame length in samples, :math:`L`.

    norm : bool
        If True, divide the zero-crossing rate by the frame length.

    softness : float > 0
        A smoothing parameter. The smaller value makes the output closer to the true
        zero-crossing rate, but the gradient vanishes.

    Returns
    -------
    out : Tensor [shape=(..., T/L)]
        The zero-crossing rate.

    """
    return nn.ZeroCrossingAnalysis._func(
        x, frame_length=frame_length, norm=norm, softness=softness
    )


def zerodf(
    x: Tensor, b: Tensor, frame_period: int = 80, ignore_gain: bool = False
) -> Tensor:
    """Apply an all-zero digital filter.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        The excitation signal.

    b : Tensor [shape=(..., T/P, M+1)]
        The filter coefficients.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    ignore_gain : bool
        If True, perform filtering without the gain.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        The output signal.

    """
    return nn.AllZeroDigitalFilter._func(
        x, b, frame_period=frame_period, ignore_gain=ignore_gain
    )
