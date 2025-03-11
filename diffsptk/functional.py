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

from . import modules as nn


def acorr(x, acr_order, out_format="naive"):
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


def acr2csm(r):
    """Convert autocorrelation to CSM coefficients.

    Parameters
    ----------
    r : Tensor [shape=(..., M+1)]
        Autocorrelation.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        CSM coefficients.

    """
    return nn.AutocorrelationToCompositeSinusoidalModelCoefficients._func(r)


def alaw(x, abs_max=1, a=87.6):
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


def b2mc(b, alpha=0):
    """Convert MLSA filter coefficients to mel-cepstrum.

    Parameters
    ----------
    b : Tensor [shape=(..., M+1)]
        MLSA filter coefficients.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Mel-cepstral coefficients.

    """
    return nn.MLSADigitalFilterCoefficientsToMelCepstrum._func(b, alpha=alpha)


def c2acr(c, acr_order, n_fft=512):
    """Convert cepstrum to autocorrelation.

    Parameters
    ----------
    c : Tensor [shape=(..., M+1)]
        Cepstral coefficients.

    acr_order : int >= 0
        Order of autocorrelation, :math:`N`.

    n_fft : int >> N
        Number of FFT bins. Accurate conversion requires the large value.

    Returns
    -------
    out : Tensor [shape=(..., N+1)]
        Autocorrelation.

    """
    return nn.CepstrumToAutocorrelation._func(c, acr_order=acr_order, n_fft=n_fft)


def c2mpir(c, ir_length, n_fft=512):
    """Convert cepstrum to minimum phase impulse response.

    Parameters
    ----------
    c : Tensor [shape=(..., M+1)]
        Cepstral coefficients.

    ir_length : int >= 1
        Length of impulse response, :math:`N`.

    n_fft : int >> N
        Number of FFT bins. Accurate conversion requires the large value.

    Returns
    -------
    out : Tensor [shape=(..., N)]
        Truncated minimum phase impulse response.

    """
    return nn.CepstrumToMinimumPhaseImpulseResponse._func(
        c, ir_length=ir_length, n_fft=n_fft
    )


def c2ndps(c, fft_length):
    """Convert cepstrum to NDPS.

    Parameters
    ----------
    c : Tensor [shape=(..., M+1)]
        Cepstrum.

    fft_length : int >= 2
        Number of FFT bins, :math:`L`.

    Returns
    -------
    out : Tensor [shape=(..., L/2+1)]
        NDPS.

    """
    return nn.CepstrumToNegativeDerivativeOfPhaseSpectrum._func(
        c, fft_length=fft_length
    )


def cdist(c1, c2, full=False, reduction="mean"):
    """Calculate cepstral distance between two inputs.

    Parameters
    ----------
    c1 : Tensor [shape=(..., M+1)]
        Input cepstral coefficients.

    c2 : Tensor [shape=(..., M+1)]
        Target cepstral coefficients.

    full : bool
        If True, include the constant term in the distance calculation.

    reduction : ['none', 'mean', 'batchmean', 'sum']
        Reduction type.

    Returns
    -------
    out : Tensor [shape=(...,) or scalar]
        Cepstral distance.

    """
    return nn.CepstralDistance._func(c1, c2, full=full, reduction=reduction)


def chroma(x, n_channel, sample_rate, norm=float("inf")):
    """Apply chroma-filter banks to STFT.

    Parameters
    ----------
    x : Tensor [shape=(..., L/2+1)]
        Power spectrum.

    n_channel : int >= 1
        Number of chroma-filter banks, :math:`C`.

    sample_rate : int >= 1
        Sample rate in Hz.

    norm : float
        Normalization factor.

    Returns
    -------
    out : Tensor [shape=(..., C)]
        Chroma-filter bank output.

    """
    return nn.ChromaFilterBankAnalysis._func(
        x,
        n_channel=n_channel,
        sample_rate=sample_rate,
        norm=norm,
        use_power=True,
    )


def csm2acr(c):
    """Convert CSM coefficients to autocorrelation.

    Parameters
    ----------
    c : Tensor [shape=(..., M+1)]
        CSM coefficients.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Autocorrelation.

    """
    return nn.CompositeSinusoidalModelCoefficientsToAutocorrelation._func(c)


def dct(x, dct_type=2):
    """Compute DCT.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        Input signal.

    dct_type : int in [1, 4]
        DCT type.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        DCT output.

    """
    return nn.DiscreteCosineTransform._func(x, dct_type=dct_type)


def decimate(x, period=1, start=0, dim=-1):
    """Decimate signal.

    Parameters
    ----------
    x : Tensor [shape=(..., T, ...)]
        Signal.

    period : int >= 1
        Decimation period, :math:`P`.

    start : int >= 0
        Start point, :math:`S`.

    dim : int
        Dimension along which to decimate the tensors.

    Returns
    -------
    out : Tensor [shape=(..., T/P-S, ...)]
        Decimated signal.

    """
    return nn.Decimation._func(x, period=period, start=start, dim=dim)


def delay(x, start=0, keeplen=False, dim=-1):
    """Delay signal.

    Parameters
    ----------
    x : Tensor [shape=(..., T, ...)]
        Signal.

    start : int
        Start point, :math:`S`. If negative, advance signal.

    keeplen : bool
        If True, output has the same length of input.

    dim : int
        Dimension along which to shift the tensors.

    Returns
    -------
    out : Tensor [shape=(..., T-S, ...)] or [shape=(..., T, ...)]
        Delayed signal.

    """
    return nn.Delay._func(x, start=start, keeplen=keeplen, dim=dim)


def delta(x, seed=[[-0.5, 0, 0.5]], static_out=True):
    """Compute delta components.

    Parameters
    ----------
    x : Tensor [shape=(B, T, D) or (T, D)]
        Static components.

    seed : list[list[float]] or list[int]
        Delta coefficients or width(s) of 1st (and 2nd) regression coefficients.

    static_out : bool
        If False, output only delta components.

    Returns
    -------
    out : Tensor [shape=(B, T, DxH) or (T, DxH)]
        Delta (and static) components.

    """
    return nn.Delta._func(x, seed, static_out=static_out)


def dequantize(y, abs_max=1, n_bit=8, quantizer="mid-rise"):
    """Dequantize input.

    Parameters
    ----------
    y : Tensor [shape=(...,)]
        Quantized input.

    abs_max : float > 0
        Absolute maximum value of input.

    n_bit : int >= 1
        Number of quantization bits.

    quantizer : ['mid-rise', 'mid-tread']
        Quantizer.

    Returns
    -------
    out : Tensor [shape=(...,)]
        Dequantized input.

    """
    return nn.InverseUniformQuantization._func(
        y, abs_max=abs_max, n_bit=n_bit, quantizer=quantizer
    )


def dfs(x, b=None, a=None, ir_length=None):
    """Apply an IIR digital filter.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        Input waveform.

    b : Tensor [shape=(M+1,)] or None
        Numerator coefficients.

    a : Tensor [shape=(N+1,)] or None
        Denominator coefficients.

    ir_length : int >= 1 or None
        The length of the truncated impulse response. If given, the filter is
        approximated by an FIR filter.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        Filtered waveform.

    """
    return nn.InfiniteImpulseResponseDigitalFilter._func(
        x, b=b, a=a, ir_length=ir_length
    )


def dht(x, dht_type=2):
    """Compute DHT.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        Input signal.

    dht_type : int in [1, 4]
        DHT type.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        DHT output.

    """
    return nn.DiscreteHartleyTransform._func(x, dht_type=dht_type)


def drc(
    x,
    threshold,
    ratio,
    attack_time,
    release_time,
    sample_rate,
    makeup_gain=0,
    abs_max=1,
):
    """Apply dynamic range compression.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        Input signal.

    threshold : float <= 0
        Threshold in dB.

    ratio : float > 1
        Input/output ratio.

    attack_time : float > 0
        Attack time in msec.

    release_time : float > 0
        Release time in msec.

    sample_rate : int >= 1
        Sample rate in Hz.

    makeup_gain : float >= 0
        Make-up gain in dB.

    abs_max : float > 0
        Absolute maximum value of input.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        Compressed signal.

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


def dst(x, dst_type=2):
    """Compute DST.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        Input signal.

    dst_type : int in [1, 4]
        DST type.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        DST output.

    """
    return nn.DiscreteSineTransform._func(x, dst_type=dst_type)


def entropy(p, out_format="nat"):
    """Calculate entropy.

    Parameters
    ----------
    p : Tensor [shape=(..., N)]
        Probability.

    out_format : ['bit', 'nat', 'dit']
        Output format.

    Returns
    -------
    out : Tensor [shape=(...,)]
        Entropy.

    """
    return nn.Entropy._func(p, out_format=out_format)


def excite(
    p,
    frame_period=80,
    *,
    voiced_region="pulse",
    unvoiced_region="gauss",
    polarity="auto",
    init_phase="zeros",
):
    """Generate a simple excitation signal.

    Parameters
    ----------
    p : Tensor [shape=(..., N)]
        Pitch in seconds.

    frame_period : int >= 1
        Frame period in samples, :math:`P`.

    voiced_region : ['pulse', 'sinusoidal', 'sawtooth', 'inverted-sawtooth', \
                     'triangle', 'square']
        Value on voiced region.

    unvoiced_region : ['gauss', 'zeros']
        Value on unvoiced region.

    polarity : ['auto', 'unipolar', 'bipolar']
        Polarity.

    init_phase : ['zeros', 'random']
        Initial phase.

    Returns
    -------
    out : Tensor [shape=(..., NxP)]
        Excitation signal.

    """
    return nn.ExcitationGeneration._func(
        p,
        frame_period=frame_period,
        voiced_region=voiced_region,
        unvoiced_region=unvoiced_region,
        polarity=polarity,
        init_phase=init_phase,
    )


def fbank(x, n_channel, sample_rate, f_min=0, f_max=None, floor=1e-5, out_format="y"):
    """Apply mel-filter banks to STFT.

    Parameters
    ----------
    x : Tensor [shape=(..., L/2+1)]
        Power spectrum.

    n_channel : int >= 1
        Number of mel-filter banks, :math:`C`.

    sample_rate : int >= 1
        Sample rate in Hz.

    f_min : float >= 0
        Minimum frequency in Hz.

    f_max : float <= sample_rate // 2
        Maximum frequency in Hz.

    floor : float > 0
        Minimum mel-filter bank output in linear scale.

    out_format : ['y', 'yE', 'y,E']
        `y` is mel-filber bank output and `E` is energy. If this is `yE`, the two output
        tensors are concatenated and return the tensor instead of the tuple.

    Returns
    -------
    y : Tensor [shape=(..., C)]
        Mel-filter bank output.

    E : Tensor [shape=(..., 1)] (optional)
        Energy.

    """
    return nn.MelFilterBankAnalysis._func(
        x,
        n_channel=n_channel,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        floor=floor,
        use_power=False,
        out_format=out_format,
    )


def fftcep(x, cep_order, accel=0, n_iter=0):
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


def flux(x, y=None, lag=1, norm=2, reduction="mean"):
    """Calculate flux.

    Parameters
    ----------
    x : Tensor [shape=(..., N, D)]
        Input.

    y : Tensor [shape=(..., N, D)] or None
        Target (optional).

    lag : int
        Lag of the distance calculation, :math:`L`.

    norm : int or float
        Order of norm.

    reduction : ['none', 'mean', 'batchmean', 'sum']
        Reduction type.

    Returns
    -------
    out : Tensor [shape=(..., N-\\|L\\|) or scalar]
        Flux.

    """
    return nn.Flux._func(x, y, lag=lag, norm=norm, reduction=reduction)


def frame(
    x, frame_length=400, frame_period=80, center=True, zmean=False, mode="constant"
):
    """Perform framing.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        Waveform.

    frame_length : int >= 1
        Frame length, :math:`L`.

    frame_peirod : int >= 1
        Frame period, :math:`P`.

    center : bool
        If True, assume that the center of data is the center of frame, otherwise
        assume that the center of data is the left edge of frame.

    zmean : bool
        If True, perform mean subtraction on each frame.

    mode : ['constant', 'reflect', 'replicate', 'circular']
        Padding mode.

    Returns
    -------
    out : Tensor [shape=(..., T/P, L)]
        Framed waveform.

    """
    return nn.Frame._func(
        x,
        frame_length=frame_length,
        frame_period=frame_period,
        center=center,
        zmean=zmean,
        mode=mode,
    )


def freqt(c, out_order, alpha=0):
    """Perform frequency transform.

    Parameters
    ----------
    c : Tensor [shape=(..., M1+1)]
        Cepstral coefficients.

    out_order : int >= 0
        Order of output cepstrum, :math:`M_2`.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    Returns
    -------
    out : Tensor [shape=(..., M2+1)]
        Warped cepstral coefficients.

    """
    return nn.FrequencyTransform._func(c, out_order=out_order, alpha=alpha)


def freqt2(c, out_order, alpha=0, theta=0, n_fft=512):
    """Perform second-order all-pass frequency transform.

    Parameters
    ----------
    c : Tensor [shape=(..., M1+1)]
        Cepstral coefficients.

    out_order : int >= 0
        Order of output cepstrum, :math:`M_2`.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    theta : float in [0, 1]
        Emphasis frequency, :math:`\\theta`.

    n_fft : int >> M2
        Number of FFT bins. Accurate conversion requires the large value.

    Returns
    -------
    out : Tensor [shape=(..., M2+1)]
        Warped cepstral coefficients.

    """
    return nn.SecondOrderAllPassFrequencyTransform._func(
        c, out_order=out_order, alpha=alpha, theta=theta, n_fft=n_fft
    )


def gnorm(x, gamma=0, c=None):
    """Perform cepstrum gain normalization.

    Parameters
    ----------
    x : Tensor [shape=(..., M+1)]
        Generalized cepstrum.

    gamma : float in [-1, 1]
        Gamma, :math:`\\gamma`.

    c : int >= 1 or None
        Number of stages.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Normalized generalized cepstrum.

    """
    return nn.GeneralizedCepstrumGainNormalization._func(x, gamma=gamma, c=c)


def grpdelay(b=None, a=None, *, fft_length=512, alpha=1, gamma=1, **kwargs):
    """Compute group delay.

    Parameters
    ----------
    b : Tensor [shape=(..., M+1)] or None
        Numerator coefficients.

    a : Tensor [shape=(..., N+1)] or None
        Denominator coefficients.

    fft_length : int >= 2
        Number of FFT bins, :math:`L`.

    alpha : float > 0
        Tuning parameter, :math:`\\alpha`.

    gamma : float > 0
        Tuning parameter, :math:`\\gamma`.

    Returns
    -------
    out : Tensor [shape=(..., L/2+1)]
        Group delay or modified group delay function.

    """
    return nn.GroupDelay._func(
        b, a, fft_length=fft_length, alpha=alpha, gamma=gamma, **kwargs
    )


def hilbert(x, fft_length=None, dim=-1):
    """Compute analytic signal using the Hilbert transform.

    Parameters
    ----------
    x : Tensor [shape=(..., T, ...)]
        Input signal.

    fft_length : int >= 1 or None
        Number of FFT bins. If None, set to :math:`T`.

    dim : int
        Dimension along which to take the Hilbert transform.

    Returns
    -------
    out : Tensor [shape=(..., T, ...)]
        Analytic signal, where real part is the input signal and imaginary part is
        the Hilbert transform of the input signal.

    """
    return nn.HilbertTransform._func(x, fft_length=fft_length, dim=dim)


def hilbert2(x, fft_length=None, dim=(-2, -1)):
    """Compute analytic signal using the Hilbert transform.

    Parameters
    ----------
    x : Tensor [shape=(..., T1, T2, ...)]
        Input signal.

    fft_length : int, list[int], or None
        Number of FFT bins. If None, set to (:math:`T1`, :math:`T2`).

    dim : list[int]
        Dimensions along which to take the Hilbert transform.

    Returns
    -------
    out : Tensor [shape=(..., T1, T2, ...)]
        Analytic signal, where real part is the input signal and imaginary part is
        the Hilbert transform of the input signal.

    """
    return nn.TwoDimensionalHilbertTransform._func(x, fft_length=fft_length, dim=dim)


def histogram(x, n_bin=10, lower_bound=0, upper_bound=1, norm=False, softness=1e-3):
    """Compute histogram.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        Input data.

    n_bin : int >= 1
        Number of bins, :math:`K`.

    lower_bound : float < U
        Lower bound of the histogram, :math:`L`.

    upper_bound : float > L
        Upper bound of the histogram, :math:`U`.

    norm : bool
        If True, normalize the histogram.

    softness : float > 0
        A smoothing parameter. The smaller value makes the output closer to the true
        histogram, but the gradient vanishes.

    Returns
    -------
    out : Tensor [shape=(..., K)]
        Histogram in [L, U].

    """
    return nn.Histogram._func(
        x,
        n_bin=n_bin,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        norm=norm,
        softness=softness,
    )


def ialaw(y, abs_max=1, a=87.6):
    """Expand waveform by A-law algorithm.

    Parameters
    ----------
    y : Tensor [shape=(...,)]
        Compressed waveform.

    abs_max : float > 0
        Absolute maximum value of input.

    a : float >= 1
        Compression factor, :math:`A`.

    Returns
    -------
    out : Tensor [shape=(...,)]
        Waveform.

    """
    return nn.ALawExpansion._func(y, abs_max=abs_max, a=a)


def idct(y, dct_type=2):
    """Compute inverse DCT.

    Parameters
    ----------
    y : Tensor [shape=(..., L)]
        Input.

    dct_type : int in [1, 4]
        DCT type.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        Inverse DCT output.

    """
    return nn.InverseDiscreteCosineTransform._func(y, dct_type=dct_type)


def idht(y, dht_type=2):
    """Compute inverse DHT.

    Parameters
    ----------
    y : Tensor [shape=(..., L)]
        Input.

    dht_type : int in [1, 4]
        DHT type.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        Inverse DHT output.

    """
    return nn.InverseDiscreteHartleyTransform._func(y, dht_type=dht_type)


def idst(y, dst_type=2):
    """Compute inverse DST.

    Parameters
    ----------
    y : Tensor [shape=(..., L)]
        Input.

    dst_type : int in [1, 4]
        DST type.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        Inverse DST output.

    """
    return nn.InverseDiscreteSineTransform._func(y, dst_type=dst_type)


def ifreqt2(c, out_order, alpha=0, theta=0, n_fft=512):
    """Perform second-order all-pass inverse frequency transform.

    Parameters
    ----------
    c : Tensor [shape=(..., M1+1)]
        Cepstral coefficients.

    out_order : int >= 0
        Order of output cepstrum, :math:`M_2`.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    theta : float in [0, 1]
        Emphasis frequency, :math:`\\theta`.

    n_fft : int >> M1
        Number of FFT bins. Accurate conversion requires the large value.

    Returns
    -------
    out : Tensor [shape=(..., M2+1)]
        Warped cepstral coefficients.

    """
    return nn.SecondOrderAllPassInverseFrequencyTransform._func(
        c, out_order=out_order, alpha=alpha, theta=theta, n_fft=n_fft
    )


def ignorm(y, gamma=0, c=None):
    """Perform cepstrum inverse gain normalization.

    Parameters
    ----------
    y : Tensor [shape=(..., M+1)]
        Normalized generalized cepstrum.

    gamma : float in [-1, 1]
        Gamma, :math:`\\gamma`.

    c : int >= 1 or None
        Number of stages.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Generalized cepstrum.

    """
    return nn.GeneralizedCepstrumInverseGainNormalization._func(y, gamma=gamma, c=c)


def imdct(y, out_length=None, frame_length=400, window="sine"):
    """Compute inverse modified discrete cosine transform.

    Parameters
    ----------
    y : Tensor [shape=(..., 2T/L, L/2)]
        Spectrum.

    out_length : int or None
        Length of output waveform.

    frame_length : int >= 2
        Frame length, :math:`L`.

    window : ['sine', 'vorbis', 'kbd', 'rectangular']
        Window type.

    """
    return nn.InverseModifiedDiscreteCosineTransform._func(
        y, out_length=out_length, frame_length=frame_length, window=window
    )


def imdst(y, out_length=None, frame_length=400, window="sine"):
    """Compute inverse modified discrete sine transform.

    Parameters
    ----------
    y : Tensor [shape=(..., 2T/L, L/2)]
        Spectrum.

    out_length : int or None
        Length of output waveform.

    frame_length : int >= 2
        Frame length, :math:`L`.

    window : ['sine', 'vorbis', 'kbd', 'rectangular']
        Window type.

    """
    return nn.InverseModifiedDiscreteSineTransform._func(
        y, out_length=out_length, frame_length=frame_length, window=window
    )


def interpolate(x, period=1, start=0, dim=-1):
    """Interpolate signal.

    Parameters
    ----------
    x : Tensor [shape=(..., T, ...)]
        Signal.

    period : int >= 1
        Interpolation period, :math:`P`.

    start : int >= 0
        Start point, :math:`S`.

    dim : int
        Dimension along which to interpolate the tensors.

    Returns
    -------
    out : Tensor [shape=(..., TxP+S, ...)]
        Interpolated signal.

    """
    return nn.Interpolation._func(x, period=period, start=start, dim=dim)


def ipnorm(y):
    """Perform cepstrum inverse power normalization.

    Parameters
    ----------
    y : Tensor [shape=(..., M+2)]
        Power-normalized cepstrum.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Output cepstrum.

    """
    return nn.MelCepstrumInversePowerNormalization._func(y)


def is2par(s):
    """Convert IS to PARCOR.

    Parameters
    ----------
    s : Tensor [shape=(..., M+1)]
        Inverse sine coefficients.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        PARCOR coefficients.

    """
    return nn.InverseSineToParcorCoefficients._func(s)


def istft(
    y,
    *,
    out_length=None,
    frame_length=400,
    frame_period=80,
    fft_length=512,
    center=True,
    window="blackman",
    norm="power",
):
    """Compute inverse short-time Fourier transform.

    Parameters
    ----------
    y : Tensor [shape=(..., T/P, N/2+1)]
        Complex spectrum.

    out_length : int >= 1 or None
        Length of output signal.

    frame_length : int >= 1
        Frame length, :math:`L`.

    frame_peirod : int >= 1
        Frame period, :math:`P`.

    fft_length : int >= L
        Number of FFT bins, :math:`N`.

    center : bool
        If True, assume that the center of data is the center of frame, otherwise
        assume that the center of data is the left edge of frame.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular', 'nuttall']
        Window type.

    norm : ['none', 'power', 'magnitude']
        Normalization type of window.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        Waveform.

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
    )


def iulaw(y, abs_max=1, mu=255):
    """Expand waveform by :math:`\\mu`-law algorithm.

    Parameters
    ----------
    y : Tensor [shape=(...,)]
        Compressed waveform.

    abs_max : float > 0
        Absolute maximum value of input.

    mu : int >= 1
        Compression factor, :math:`\\mu`.

    Returns
    -------
    out : Tensor [shape=(...,)]
        Waveform.

    """
    return nn.MuLawExpansion._func(y, abs_max=abs_max, mu=mu)


def iwht(y, wht_type="natural"):
    """Compute inverse WHT.

    Parameters
    ----------
    y : Tensor [shape=(..., L)]
        Input.

    wht_type : ['sequency', 'natural', 'dyadic']
        Order of coefficients of Walsh matrix.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        Inverse WHT output.

    """
    return nn.InverseWalshHadamardTransform._func(y, wht_type=wht_type)


def lar2par(g):
    """Convert LAR to PARCOR.

    Parameters
    ----------
    g : Tensor [shape=(..., M+1)]
        Log area ratio.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        PARCOR coefficients.

    """
    return nn.LogAreaRatioToParcorCoefficients._func(g)


def levdur(r, eps=0):
    """Solve a Yule-Walker linear system.

    Parameters
    ----------
    r : Tensor [shape=(..., M+1)]
        Autocorrelation.

    eps : float >= 0
        A small value to improve numerical stability.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Gain and LPC coefficients.

    """
    return nn.LevinsonDurbin._func(r, eps=eps)


def linear_intpl(x, upsampling_factor=80):
    """Interpolate filter coefficients.

    Parameters
    ----------
    x : Tensor [shape=(B, N, D) or (N, D) or (N,)]
        Filter coefficients.

    upsampling_factor : int >= 1
        Upsampling factor, :math:`P`.

    Returns
    -------
    y : Tensor [shape=(B, NxP, D) or (NxP, D) or (NxP,)]
        Upsampled filter coefficients.

    """
    return nn.LinearInterpolation._func(x, upsampling_factor=upsampling_factor)


def lpc(x, lpc_order, eps=1e-6):
    """Compute LPC coefficients.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        Framed waveform.

    lpc_order : int >= 0
        Order of LPC, :math:`M`.

    eps : float >= 0
        A small value to improve numerical stability.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Gain and LPC coefficients.

    """
    return nn.LinearPredictiveCodingAnalysis._func(x, lpc_order=lpc_order, eps=eps)


def lpc2lsp(a, log_gain=False, sample_rate=None, out_format="radian"):
    """Convert LPC to LSP.

    Parameters
    ----------
    a : Tensor [shape=(..., M+1)]
        LPC coefficients.

    log_gain : bool
        If True, output gain in log scale.

    sample_rate : int >= 1 or None
        Sample rate in Hz.

    out_format : ['radian', 'cycle', 'khz', 'hz']
        Output format.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        LSP frequencies.

    """
    return nn.LinearPredictiveCoefficientsToLineSpectralPairs._func(
        a, log_gain=log_gain, sample_rate=sample_rate, out_format=out_format
    )


def lpc2par(a, gamma=1, c=None):
    """Convert LPC to PARCOR.

    Parameters
    ----------
    a : Tensor [shape=(..., M+1)]
        LPC coefficients.

    gamma : float in [-1, 1]
        Gamma, :math:`\\gamma`.

    c : int >= 1 or None
        Number of stages.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        PARCOR coefficients.

    """
    return nn.LinearPredictiveCoefficientsToParcorCoefficients._func(
        a, gamma=gamma, c=c
    )


def lpccheck(a, margin=1e-16, warn_type="warn"):
    """Check stability of LPC coefficients.

    Parameters
    ----------
    a : Tensor [shape=(..., M+1)]
        LPC coefficients.

    margin : float in (0, 1)
        Margin for stability.

    warn_type : ['ignore', 'warn', 'exit']
        Warning type.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Modified LPC coefficients.

    """
    return nn.LinearPredictiveCoefficientsStabilityCheck._func(
        a, margin=margin, warn_type=warn_type
    )


def lsp2lpc(w, log_gain=False, sample_rate=None, in_format="radian"):
    """Convert LSP to LPC.

    Parameters
    ----------
    w : Tensor [shape=(..., M+1)]
        LSP frequencies in radians.

    log_gain : bool
        If True, assume input gain is in log scale.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        LPC coefficients.

    """
    return nn.LineSpectralPairsToLinearPredictiveCoefficients._func(
        w, log_gain=log_gain, sample_rate=sample_rate, in_format=in_format
    )


def lspcheck(w, rate=0, n_iter=1, warn_type="warn"):
    """Check stability of LSP frequencies.

    Parameters
    ----------
    w : Tensor [shape=(..., M+1)]
        LSP frequencies in radians.

    rate : float in [0, 1]
        Rate of distance between two adjacent LSPs.

    n_iter : int >= 0
        Number of iterations for modification.

    warn_type : ['ignore', 'warn', 'exit']
        Warning type.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Modified LSP frequencies.

    """
    return nn.LineSpectralPairsStabilityCheck._func(
        w, rate=rate, n_iter=n_iter, warn_type=warn_type
    )


def lsp2sp(w, fft_length, alpha=0, gamma=-1, log_gain=False, out_format="power"):
    """Convert line spectral pairs to spectrum.

    Parameters
    ----------
    w : Tensor [shape=(..., M+1)]
        Line spectral pairs in radians.

    fft_length : int >= 1
        Number of FFT bins, :math:`L`.

    alpha : float in (-1, 1)
        Warping factor, :math:`\\alpha`.

    gamma : float in [-1, 0)
        Gamma, :math:`\\gamma`.

    log_gain : bool
        If True, assume input gain is in log scale.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        Output format.

    Returns
    -------
    out : Tensor [shape=(..., L/2+1)]
        Spectrum.

    """
    return nn.LineSpectralPairsToSpectrum._func(
        w,
        fft_length=fft_length,
        alpha=alpha,
        gamma=gamma,
        log_gain=log_gain,
        out_format=out_format,
    )


def magic_intpl(x, magic_number=0):
    """Interpolate magic number.

    Parameters
    ----------
    x : Tensor [shape=(B, N, D) or (N, D) or (N,)]
        Data containing magic number.

    magic_number : float or Tensor
        Magic number.

    Returns
    -------
    out : Tensor [shape=(B, N, D) or (N, D) or (N,)]
        Data after interpolation.

    """
    return nn.MagicNumberInterpolation._func(x, magic_number=magic_number)


def mc2b(mc, alpha=0):
    """Convert mel-cepstrum to MLSA digital filter coefficients.

    Parameters
    ----------
    mc : Tensor [shape=(..., M+1)]
        Mel-cepstral coefficients.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        MLSA digital filter coefficients.

    """
    return nn.MelCepstrumToMLSADigitalFilterCoefficients._func(mc, alpha=alpha)


def mcep(x, cep_order, alpha=0, n_iter=0):
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


def mcpf(mc, alpha=0, beta=0, onset=2, ir_length=128):
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


def mdct(x, frame_length=400, window="sine"):
    """Compute modified discrete cosine transform.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        Waveform.

    frame_length : int >= 2
        Frame length, :math:`L`.

    window : ['sine', 'vorbis', 'kbd', 'rectangular']
        Window type.

    Returns
    -------
    out : Tensor [shape=(..., 2T/L, L/2)]
        Spectrum.

    """
    return nn.ModifiedDiscreteCosineTransform._func(
        x, frame_length=frame_length, window=window
    )


def mdst(x, frame_length=400, window="sine"):
    """Compute modified discrete sine transform.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        Waveform.

    frame_length : int >= 2
        Frame length, :math:`L`.

    window : ['sine', 'vorbis', 'kbd', 'rectangular']
        Window type.

    Returns
    -------
    out : Tensor [shape=(..., 2T/L, L/2)]
        Spectrum.

    """
    return nn.ModifiedDiscreteSineTransform._func(
        x, frame_length=frame_length, window=window
    )


def mfcc(
    x,
    mfcc_order,
    n_channel,
    sample_rate,
    lifter=1,
    f_min=0,
    f_max=None,
    floor=1e-5,
    out_format="y",
):
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
        out_format=out_format,
    )


def mgc2mgc(
    mc,
    out_order,
    in_alpha=0,
    out_alpha=0,
    in_gamma=0,
    out_gamma=0,
    in_norm=False,
    out_norm=False,
    in_mul=False,
    out_mul=False,
    n_fft=512,
):
    """Convert mel-generalized cepstrum to mel-generalized cepstrum.

    Parameters
    ----------
    mc : Tensor [shape=(..., M1+1)]
        Input mel-cepstrum.

    out_order : int >= 0
        Order of output cepstrum, :math:`M_2`.

    in_alpha : float in (-1, 1)
        Input alpha, :math:`\\alpha_1`.

    out_alpha : float in (-1, 1)
        Output alpha, :math:`\\alpha_2`.

    in_gamma : float in [-1, 1]
        Input gamma, :math:`\\gamma_1`.

    out_gamma : float in [-1, 1]
        Output gamma, :math:`\\gamma_2`.

    in_norm : bool
        If True, assume normalized input.

    out_norm : bool
        If True, assume normalized output.

    in_mul : bool
        If True, assume gamma-multiplied input.

    out_mul : bool
        If True, assume gamma-multiplied output.

    n_fft : int >> M1, M2
        Number of FFT bins. Accurate conversion requires the large value.

    Returns
    -------
    out : Tensor [shape=(..., M2+1)]
        Converted mel-cepstrum.

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
    mc,
    fft_length,
    alpha=0,
    gamma=0,
    norm=False,
    mul=False,
    n_fft=512,
    out_format="power",
):
    """Convert mel-cepstrum to spectrum.

    Parameters
    ----------
    mc : Tensor [shape=(..., M+1)]
        Mel-cepstrum.

    fft_length : int >= 2
        Number of FFT bins, :math:`L`.

    alpha : float in (-1, 1)
        Warping factor, :math:`\\alpha`.

    gamma : float in [-1, 1]
        Gamma, :math:`\\gamma`.

    norm : bool
        If True, assume normalized cepstrum.

    mul : bool
        If True, assume gamma-multiplied cepstrum.

    n_fft : int >> L
        Number of FFT bins. Accurate conversion requires the large value.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power', \
                  'cycle', 'radian', 'degree', 'complex']
        Output format.

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


def mlpg(u, seed=[[-0.5, 0, 0.5], [1, -2, 1]]):
    """Perform MLPG to obtain smoothed static sequence.

    Parameters
    ----------
    u : Tensor [shape=(..., T, DxH)]
        Time-variant mean vectors with delta components.

    seed : list[list[float]] or list[int]
        Delta coefficients or width(s) of 1st (and 2nd) regression coefficients.

    Returns
    -------
    out : Tensor [shape=(..., T, D)]
        Static components.

    """
    return nn.MaximumLikelihoodParameterGeneration._func(u, seed=seed)


def mlsacheck(
    c,
    *,
    alpha=0,
    pade_order=4,
    strict=True,
    threshold=None,
    fast=True,
    n_fft=512,
    warn_type="warn",
    mod_type="scale",
):
    """Check stability of MLSA filter.

    Parameters
    ----------
    c : Tensor [shape=(..., M+1)]
        Mel-cepstrum.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    pade_order : int in [4, 7]
        Order of Pade approximation.

    strict : bool
        If True, keep maximum log approximation error rather than MLSA filter stability.

    threshold : float > 0 or None
        Threshold value. If not given, automatically computed.

    fast : bool
        Enable fast mode.

    n_fft : int > M
        Number of FFT bins, :math:`L`. Used only in non-fast mode.

    warn_type : ['ignore', 'warn', 'exit']
        Warning type.

    mod_type : ['clip', 'scale']
        Modification type.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Modified mel-cepstrum.

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


def mpir2c(h, cep_order, n_fft=512):
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


def ndps2c(n, cep_order):
    """Convert NPDS to cepstrum.

    Parameters
    ----------
    n : Tensor [shape=(..., L/2+1)]
        NDPS, where :math:`L` is the number of FFT bins.

    cep_order : int >= 0
        Order of cepstrum, :math:`M`.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Cepstrum.

    """
    return nn.NegativeDerivativeOfPhaseSpectrumToCepstrum._func(n, cep_order=cep_order)


def norm0(a):
    """Convert all-pole to all-zero filter coefficients vice versa.

    Parameters
    ----------
    a : Tensor [shape=(..., M+1)]
        All-pole or all-zero filter coefficients.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        All-zero or all-pole filter coefficients.

    """
    return nn.AllPoleToAllZeroDigitalFilterCoefficients._func(a)


def par2is(k):
    """Convert PARCOR to IS.

    Parameters
    ----------
    k : Tensor [shape=(..., M+1)]
        PARCOR coefficients.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Inverse sine coefficients.

    """
    return nn.ParcorCoefficientsToInverseSine._func(k)


def par2lar(k):
    """Convert PARCOR to LAR.

    Parameters
    ----------
    k : Tensor [shape=(..., M+1)]
        PARCOR coefficients.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Log area ratio.

    """
    return nn.ParcorCoefficientsToLogAreaRatio._func(k)


def par2lpc(k, gamma=1, c=None):
    """Convert PARCOR to LPC.

    Parameters
    ----------
    k : Tensor [shape=(..., M+1)]
        PARCOR coefficients.

    gamma : float in [-1, 1]
        Gamma, :math:`\\gamma`.

    c : int >= 1 or None
        Number of stages.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        LPC coefficients.

    """
    return nn.ParcorCoefficientsToLinearPredictiveCoefficients._func(
        k, gamma=gamma, c=c
    )


def phase(b=None, a=None, *, fft_length=512, unwrap=False):
    """Compute phase spectrum.

    Parameters
    ----------
    b : Tensor [shape=(..., M+1)] or None
        Numerator coefficients.

    a : Tensor [shape=(..., N+1)] or None
        Denominator coefficients.

    fft_length : int >= 2
        Number of FFT bins, :math:`L`.

    unwrap : bool
        If True, perform phase unwrapping.

    Returns
    -------
    out : Tensor [shape=(..., L/2+1)]
        Phase spectrum [:math:`\\pi` rad].

    """
    return nn.Phase._func(b, a, fft_length=fft_length, unwrap=unwrap)


def plp(
    x,
    plp_order,
    n_channel,
    sample_rate,
    compression_factor=0.33,
    lifter=1,
    f_min=0,
    f_max=None,
    floor=1e-5,
    n_fft=512,
    out_format="y",
):
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

    n_fft : int >> M
        The number of FFT bins for the conversion from LPC to cepstrum.
        The accurate conversion requires the large value.

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
        n_fft=n_fft,
        out_format=out_format,
    )


def pnorm(x, alpha=0, ir_length=128):
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


def pol_root(x, *, eps=None, in_format="rectangular"):
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


def poledf(x, a, frame_period=80, ignore_gain=False):
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


def quantize(x, abs_max=1, n_bit=8, quantizer="mid-rise"):
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


def rlevdur(a):
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


def rmse(x, y, reduction="mean"):
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


def root_pol(a, *, eps=None, out_format="rectangular"):
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


def smcep(x, cep_order, alpha=0, theta=0, n_iter=0, accuracy_factor=4):
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


def snr(s, sn, frame_length=None, full=False, reduction="mean", eps=1e-8):
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
    b=None, a=None, *, fft_length=512, eps=0, relative_floor=None, out_format="power"
):
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
    x,
    *,
    frame_length=400,
    frame_period=80,
    fft_length=512,
    center=True,
    zmean=False,
    mode="constant",
    window="blackman",
    norm="power",
    eps=1e-9,
    relative_floor=None,
    out_format="power",
):
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
        eps=eps,
        relative_floor=relative_floor,
        out_format=out_format,
    )


def ulaw(x, abs_max=1, mu=255):
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
    y,
    *,
    out_length=None,
    frame_period=80,
    center=True,
    window="rectangular",
    norm="none",
):
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
    )


def wht(x, wht_type="natural"):
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


def window(x, out_length=None, *, window="blackman", norm="power"):
    """Apply a window function to the given waveform.

    Parameters
    ----------
    x : Tensor [shape=(..., L1)]
        The input framed waveform.

    out_length : int >= L1 or None
        The output length, :math:`L_2`. If :math:`L_2 > L_1`, output is zero-padded.
        If None, :math:`L_2 = L_1`.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular', 'nuttall']
        The window type.

    norm : ['none', 'power', 'magnitude']
        The normalization type of the window.

    Returns
    -------
    out : Tensor [shape=(..., L2)]
        The windowed waveform.

    """
    return nn.Window._func(x, out_length=out_length, window=window, norm=norm)


def yingram(x, sample_rate=22050, lag_min=22, lag_max=None, n_bin=20):
    """Pitch-related feature extraction module based on YIN.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        Framed waveform.

    sample_rate : int >= 1
        Sample rate in Hz.

    lag_min : int >= 1
        Minimum lag in points.

    lag_max : int <= :math:`L` or None
        Maximum lag in points.

    n_bin : int >= 1
        Number of bins of Yingram to represent a semitone range.

    Returns
    -------
    out : Tensor [shape=(..., M)]
        Yingram.

    """
    return nn.Yingram._func(
        x, sample_rate=sample_rate, lag_min=lag_min, lag_max=lag_max, n_bin=n_bin
    )


def zcross(x, frame_length, norm=False, softness=1e-3):
    """Compute zero-crossing rate.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        Waveform.

    frame_length : int >= 1
        Frame length, :math:`L`.

    norm : bool
        If True, divide zero-crossing rate by frame length.

    softness : float > 0
        A smoothing parameter. The smaller value makes the output closer to the true
        zero-crossing rate, but the gradient vanishes.

    Returns
    -------
    out : Tensor [shape=(..., T/L)]
        Zero-crossing rate.

    """
    return nn.ZeroCrossingAnalysis._func(
        x, frame_length=frame_length, norm=norm, softness=softness
    )


def zerodf(x, b, frame_period=80, ignore_gain=False):
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
