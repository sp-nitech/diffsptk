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


def acorr(x, acr_order, norm=False, estimator="none"):
    """Compute autocorrelation.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        Framed waveform.

    acr_order : int >= 0
        Order of autocorrelation, :math:`M`.

    norm : bool
        If True, normalize the autocorrelation.

    estimator : ['none', 'biased', 'unbiased']
        Estimator of autocorrelation.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Autocorrelation.

    """
    return nn.Autocorrelation._func(
        x, acr_order=acr_order, norm=norm, estimator=estimator
    )


def alaw(x, abs_max=1, a=87.6):
    """Compress waveform by A-law algorithm.

    Parameters
    ----------
    x : Tensor [shape=(...,)]
        Waveform.

    abs_max : float > 0
        Absolute maximum value of input.

    a : float >= 1
        Compression factor, :math:`A`.

    Returns
    -------
    out : Tensor [shape=(...,)]
        Compressed waveform.

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


def cdist(c1, c2, full=False, reduction="mean", eps=1e-8):
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

    eps : float >= 0
        A small value to prevent NaN.

    Returns
    -------
    out : Tensor [shape=(...,) or scalar]
        Cepstral distance.

    """
    return nn.CepstralDistance._func(c1, c2, full=full, reduction=reduction, eps=eps)


def dct(x):
    """Compute DCT.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        Input signal.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        DCT output.

    """
    return nn.DiscreteCosineTransform._func(x)


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


def dfs(x, b=None, a=None):
    """Apply an IIR digital filter.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        Input waveform.

    b : Tensor [shape=(M+1,)] or None
        Numerator coefficients.

    a : Tensor [shape=(N+1,)] or None
        Denominator coefficients.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        Filtered waveform.

    """
    return nn.InfiniteImpulseResponseDigitalFilter._func(x, b=b, a=a)


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


def excite(p, frame_period=80, voiced_region="pulse", unvoiced_region="gauss"):
    """Generate a simple excitation signal.

    Parameters
    ----------
    p : Tensor [shape=(..., N)]
        Pitch in seconds.

    frame_period : int >= 1
        Frame period in samples, :math:`P`.

    voiced_region : ['pulse', 'sinusoidal', 'sawtooth']
        Value on voiced region.

    unvoiced_region : ['gauss', 'zeros']
        Value on unvoiced region.

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
        `y` is mel-filber bank outpus and `E` is energy. If this is `yE`, the two output
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


def fftcep(x, cep_order, n_iter=0, accel=0):
    """Estimate cepstrum from spectrum.

    Parameters
    ----------
    x : Tensor [shape=(..., L/2+1)]
        Power spectrum.

    cep_order : int >= 0
        Order of cepstrum, :math:`M`.

    n_iter : int >= 0
        Number of iterations.

    accel : float >= 0
        Acceleration factor.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Cepstrum.

    """
    return nn.CepstralAnalysis._func(x, cep_order=cep_order, n_iter=n_iter, accel=accel)


def frame(x, frame_length=400, frame_period=80, center=True, zmean=False):
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
        Frquency warping factor, :math:`\\alpha`.

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


def idct(y):
    """Compute inverse DCT.

    Parameters
    ----------
    y : Tensor [shape=(..., L)]
        Input.

    Returns
    -------
    out : Tensor [shape=(..., L)]
        Inverse DCT output.

    """
    return nn.InverseDiscreteCosineTransform._func(y)


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
        'rectangular']
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


def levdur(r):
    """Solve a Yule-Walker linear system.

    Parameters
    ----------
    r : Tensor [shape=(..., M+1)]
        Autocorrelation.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Gain and LPC coefficients.

    """
    return nn.LevinsonDurbin._func(r)


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


def lpc(x, lpc_order):
    """Compute LPC coefficients.

    Parameters
    ----------
    x : Tensor [shape=(..., L)]
        Framed waveform.

    lpc_order : int >= 0
        Order of LPC, :math:`M`.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Gain and LPC coefficients.

    """
    return nn.LinearPredictiveCodingAnalysis._func(x, lpc_order=lpc_order)


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


def lsp2lpc(w, log_gain=False):
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
        w, log_gain=log_gain
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
    """Convert minimum phase impulse response to cepstrum.

    Parameters
    ----------
    h : Tensor [shape=(..., N)]
        Minimum phase impulse response.

    cep_order : int >= 0
        Order of cepstrum, :math:`M`.

    n_fft : int >> N
        Number of FFT bins. Accurate conversion requires the large value.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Cepstrum.

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


def pol_root(x, real=False):
    """Compute polynomial coefficients from roots.

    Parameters
    ----------
    x : Tensor [shape=(..., M)]
        Complex roots.

    real : bool
        If True, return as real numbers.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Polynomial coefficients.

    """
    return nn.RootsToPolynomial._func(x, real=real)


def poledf(x, a, frame_period=80, ignore_gain=False):
    """Apply an all-pole digital filter.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        Excitation signal.

    a : Tensor [shape=(..., T/P, M+1)]
        Filter coefficients.

    frame_period : int >= 1
        Frame period, :math:`P`.

    ignore_gain : bool
        If True, perform filtering without gain.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        Output signal.

    """
    return nn.AllPoleDigitalFilter._func(
        x, a, frame_period=frame_period, ignore_gain=ignore_gain
    )


def quantize(x, abs_max=1, n_bit=8, quantizer="mid-rise"):
    """Quantize input.

    Parameters
    ----------
    x : Tensor [shape=(...,)]
        Input.

    abs_max : float > 0
        Absolute maximum value of input.

    n_bit : int >= 1
        Number of quantization bits.

    quantizer : ['mid-rise', 'mid-tread']
        Quantizer.

    Returns
    -------
    out : Tensor [shape=(...,)]
        Quantized input.

    """
    return nn.UniformQuantization._func(
        x, abs_max=abs_max, n_bit=n_bit, quantizer=quantizer
    )


def rlevdur(a):
    """Solve a Yule-Walker linear system given LPC coefficients.

    Parameters
    ----------
    a : Tensor [shape=(..., M+1)]
        Gain and LPC coefficients.

    Returns
    -------
    out : Tensor [shape=(..., M+1)]
        Autocorrelation.

    """
    return nn.ReverseLevinsonDurbin._func(a)


def rmse(x, y, reduction="mean", eps=1e-8):
    """Calculate RMSE.

    Parameters
    ----------
    x : Tensor [shape=(...,)]
        Input.

    y : Tensor [shape=(...,)]
        Target.

    reduction : ['none', 'mean', 'sum']
        Reduction type.

    eps : float >= 0
        A small value to prevent NaN.

    Returns
    -------
    out : Tensor [shape=(...,) or scalar]
        RMSE.

    """
    return nn.RootMeanSquareError._func(x, y, reduction=reduction, eps=eps)


def root_pol(a, out_format="rectangular"):
    """Compute roots of polynomial.

    Parameters
    ----------
    a : Tensor [shape=(..., M+1)]
        Polynomial coefficients.

    out_format : ['rectangular', 'polar']
        Output format.

    Returns
    -------
    out : Tensor [shape=(..., M)]
        Roots of polynomial.

    """
    return nn.PolynomialToRoots._func(a, out_format=out_format)


def snr(s, sn, frame_length=None, full=False, reduction="mean", eps=1e-8):
    """Calculate SNR.

    Parameters
    ----------
    s : Tensor [shape=(...,)]
        Signal.

    sn : Tensor [shape=(...,)]
        Signal plus noise.

    frame_length : int >= 1 or None
        Frame length, :math:`L`. If given, calculate segmental SNR.

    full : bool
        If True, include the constant term in the SNR calculation.

    reduction : ['none', 'mean', 'sum']
        Reduction type.

    eps : float >= 0
        A small value to prevent NaN.

    Returns
    -------
    out : Tensor [shape=(...,) or scalar]
        Signal-to-noise ratio.

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
        Numerator coefficients.

    a : Tensor [shape=(..., N+1)] or None
        Denominator coefficients.

    fft_length : int >= 2
        Number of FFT bins, :math:`L`.

    eps : float >= 0
        A small value added to power spectrum.

    relative_floor : float < 0 or None
        Relative floor in decibels.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        Output format.

    Returns
    -------
    out : Tensor [shape=(..., L/2+1)]
        Spectrum.

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
        Waveform.

    frame_length : int >= 1
        Frame length, :math:`L`.

    frame_peirod : int >= 1
        Frame period, :math:`P`.

    fft_length : int >= L
        Number of FFT bins, :math:`N`.

    center : bool
        If True, assume that the center of data is the center of frame, otherwise
        assume that the center of data is the left edge of frame.

    zmean : bool
        If True, perform mean subtraction on each frame.


    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
        'rectangular']
        Window type.

    norm : ['none', 'power', 'magnitude']
        Normalization type of window.

    eps : float >= 0
        A small value added to power spectrum.

    relative_floor : float < 0 or None
        Relative floor in decibels.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power', 'complex']
        Output format.

    Returns
    -------
    out : Tensor [shape=(..., T/P, N/2+1)]
        Spectrum.

    """
    return nn.ShortTimeFourierTransform._func(
        x,
        frame_length=frame_length,
        frame_period=frame_period,
        fft_length=fft_length,
        center=center,
        zmean=zmean,
        window=window,
        norm=norm,
        eps=eps,
        relative_floor=relative_floor,
        out_format=out_format,
    )


def ulaw(x, abs_max=1, mu=255):
    """Compress waveform by :math:`\\mu`-law algorithm.

    Parameters
    ----------
    x : Tensor [shape=(...,)]
        Waveform.

    abs_max : float > 0
        Absolute maximum value of input.

    mu : int >= 1
        Compression factor, :math:`\\mu`.

    Returns
    -------
    out : Tensor [shape=(...,)]
        Compressed waveform.

    """
    return nn.MuLawCompression._func(x, abs_max=abs_max, mu=mu)


def unframe(
    y,
    *,
    out_length=None,
    frame_length=400,
    frame_period=80,
    center=True,
    window="rectangular",
    norm="none",
):
    """Revert framed waveform.

    Parameters
    ----------
    y : Tensor [shape=(..., T/P, L)]
        Framed waveform.

    out_length : int >= 1 or None
        Length of original signal, `T`.

    frame_length : int >= 1
        Frame length, :math:`L`.

    frame_peirod : int >= 1
        Frame period, :math:`P`.

    center : bool
        If True, assume that the center of data is the center of frame, otherwise
        assume that the center of data is the left edge of frame.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular']
        Window type.

    norm : ['none', 'power', 'magnitude']
        Normalization type of window.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        Waveform.

    """
    return nn.Unframe._func(
        y,
        out_length=out_length,
        frame_length=frame_length,
        frame_period=frame_period,
        center=center,
        window=window,
        norm=norm,
    )


def window(x, out_length=None, *, window="blackman", norm="power"):
    """Apply window function.

    Parameters
    ----------
    x : Tensor [shape=(..., L1)]
        Framed waveform.

    out_length : int >= L1 or None
        Output length, :math:`L_2`.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular']
        Window type.

    norm : ['none', 'power', 'magnitude']
        Normalization type of window.

    Returns
    -------
    out : Tensor [shape=(..., L2)]
        Windowed waveform.

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
        Excitation signal.

    b : Tensor [shape=(..., T/P, M+1)]
        Filter coefficients.

    frame_period : int >= 1
        Frame period, :math:`P`.

    ignore_gain : bool
        If True, perform filtering without gain.

    Returns
    -------
    out : Tensor [shape=(..., T)]
        Output signal.

    """
    return nn.AllZeroDigitalFilter._func(
        x, b, frame_period=frame_period, ignore_gain=ignore_gain
    )
