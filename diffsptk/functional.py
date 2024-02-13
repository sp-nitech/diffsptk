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
    Tensor [shape=(...,)]
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
    Tensor [shape=(..., M+1)]
        Mel-cepstral coefficients.

    """
    return nn.MLSADigitalFilterCoefficientsToMelCepstrum._func(b, alpha=alpha)


def c2acr(c, acr_order=0, n_fft=512):
    """Convert cepstrum to autocorrelation.

    Parameters
    ----------
    c : Tensor [shape=(..., M+1)]
        Cepstral coefficients.

    acr_order : int >= 0
        Order of autocorrelation, :math:`N`.

    n_fft : int >> :math:`N`
        Number of FFT bins. Accurate conversion requires the large value.

    Returns
    -------
    Tensor [shape=(..., N+1)]
        Autocorrelation.

    """
    return nn.CepstrumToAutocorrelation._func(c, acr_order=acr_order, n_fft=n_fft)


def c2mpir(c, ir_length=1, n_fft=512):
    """Convert cepstrum to minimum phase impulse response.

    Parameters
    ----------
    c : Tensor [shape=(..., M+1)]
        Cepstral coefficients.

    ir_length : int >= 1
        Length of impulse response, :math:`N`.

    n_fft : int >> :math:`N`
        Number of FFT bins. Accurate conversion requires the large value.

    Returns
    -------
    Tensor [shape=(..., N)]
        Truncated minimum phase impulse response.

    """
    return nn.CepstrumToMinimumPhaseImpulseResponse._func(
        c, ir_length=ir_length, n_fft=n_fft
    )


def c2ndps(c, fft_length=512):
    """Convert cepstrum to NDPS.

    Parameters
    ----------
    c : Tensor [shape=(..., M+1)]
        Cepstrum.

    fft_length : int >= 2
        Number of FFT bins, :math:`L`.

    Returns
    -------
    Tensor [shape=(..., L/2+1)]
        NDPS.

    """
    return nn.CepstrumToNegativeDerivativeOfPhaseSpectrum._func(
        c, fft_length=fft_length
    )


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
    Tensor [shape=(..., T/P-S, ...)]
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
    Tensor [shape=(..., T-S, ...)] or [shape=(..., T, ...)]
        Delayed signal.

    """
    return nn.Delay._func(x, start=start, keeplen=keeplen, dim=dim)


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
    Tensor [shape=(...,)]
        Dequantized input.

    """
    return nn.InverseUniformQuantization._func(
        y, abs_max=abs_max, n_bit=n_bit, quantizer=quantizer
    )


def excite(p, frame_period=1, voiced_region="pulse", unvoiced_region="gauss"):
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
    Tensor [shape=(..., NxP)]
        Excitation signal.

    """
    return nn.ExcitationGeneration._func(
        p,
        frame_period=frame_period,
        voiced_region=voiced_region,
        unvoiced_region=unvoiced_region,
    )


def fftcep(x, cep_order=0, n_iter=0, accel=0):
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
    Tensor [shape=(..., M+1)]
        Cepstrum.

    """
    return nn.CepstralAnalysis._func(x, cep_order=cep_order, n_iter=n_iter, accel=accel)


def frame(x, frame_length=1, frame_period=1, center=True, zmean=False):
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
    Tensor [shape=(..., T/P, L)]
        Framed waveform.

    """
    return nn.Frame._func(
        x,
        frame_length=frame_length,
        frame_period=frame_period,
        center=center,
        zmean=zmean,
    )


def freqt(c, out_order=None, alpha=0):
    """Perform frequency transform.

    Parameters
    ----------
    c : Tensor [shape=(..., M1+1)]
        Cepstral coefficients.

    out_order : int >= 0 or None
        Order of output cepstrum, :math:`M_2`. If None, set to :math:`M_1`.

    alpha : float in (-1, 1)
        Frquency warping factor, :math:`\\alpha`.

    Returns
    -------
    Tensor [shape=(..., M2+1)]
        Warped cepstral coefficients.

    """
    return nn.FrequencyTransform._func(c, out_order=out_order, alpha=alpha)


def gnorm(x, gamma=0, c=None):
    """Perform cepstrum gain normalization.

    Parameters
    ----------
    x : Tensor [shape=(..., M+1)]
        Generalized cepstrum.

    Parameters
    ----------
    gamma : float in [-1, 1]
        Gamma, :math:`\\gamma`.

    c : int >= 1 or None
        Number of stages.

    Returns
    -------
    Tensor [shape=(..., M+1)]
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
    Tensor [shape=(..., L/2+1)]
        Group delay or modified group delay function.

    """
    return nn.GroupDelay._func(
        b, a, fft_length=fft_length, alpha=alpha, gamma=gamma, **kwargs
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
    Tensor [shape=(...,)]
        Waveform.

    """
    return nn.ALawExpansion._func(y, abs_max=abs_max, a=a)


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
    Tensor [shape=(..., M+1)]
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
    Tensor [shape=(..., TxP+S, ...)]
        Interpolated signal.

    """
    return nn.Interpolation._func(x, period=period, start=start, dim=dim)


def istft(
    y,
    *,
    out_length=None,
    frame_length=400,
    frame_period=160,
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
    Tensor [shape=(..., T)]
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
    Tensor [shape=(...,)]
        Waveform.

    """
    return nn.MuLawExpansion._func(y, abs_max=abs_max, mu=mu)


def levdur(r):
    """Solve a Yule-Walker linear system.

    Parameters
    ----------
    r : Tensor [shape=(..., M+1)]
        Autocorrelation.

    Returns
    -------
    Tensor [shape=(..., M+1)]
        Gain and LPC coefficients.

    """
    return nn.LevinsonDurbin._func(r)


def linear_intpl(x, upsampling_factor=1):
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
    Tensor [shape=(B, N, D) or (N, D) or (N,)]
        Data after interpolation.

    Examples
    --------
    >>> x = torch.tensor([0, 1, 2, 0, 4, 0]).float()
    >>> x
    tensor([0., 1., 2., 0., 4., 0.])
    >>> magic_intpl = diffsptk.MagicNumberInterpolation(0)
    >>> y = magic_intpl(x)
    >>> y
    tensor([1., 1., 2., 3., 4., 4.])

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
    Tensor [shape=(..., M+1)]
        MLSA digital filter coefficients.

    """
    return nn.MelCepstrumToMLSADigitalFilterCoefficients._func(mc, alpha=alpha)


def mpir2c(h, cep_order=0, n_fft=512):
    """Convert minimum phase impulse response to cepstrum.

    Parameters
    ----------
    h : Tensor [shape=(..., N)]
        Minimum phase impulse response.

    cep_order : int >= 0
        Order of cepstrum, :math:`M`.

    n_fft : int >> :math:`N`
        Number of FFT bins. Accurate conversion requires the large value.

    Returns
    -------
    Tensor [shape=(..., M+1)]
        Cepstrum.

    """
    return nn.MinimumPhaseImpulseResponseToCepstrum._func(
        h, cep_order=cep_order, n_fft=n_fft
    )


def ndps2c(n, cep_order=0):
    """Convert NPDS to cepstrum.

    Parameters
    ----------
    n : Tensor [shape=(..., L/2+1)]
        NDPS, where :math:`L` is the number of FFT bins.

    cep_order : int >= 0
        Order of cepstrum, :math:`M`.

    Returns
    -------
    Tensor [shape=(..., M+1)]
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
    Tensor [shape=(..., M+1)]
        All-zero or all-pole filter coefficients.

    """
    return nn.AllPoleToAllZeroDigitalFilterCoefficients._func(a)


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
    Tensor [shape=(..., L/2+1)]
        Phase spectrum [:math:`\\pi` rad].

    """
    return nn.Phase._func(b, a, fft_length=fft_length, unwrap=unwrap)


def pol_root(x):
    """Compute polynomial coefficients from roots.

    Parameters
    ----------
    x : Tensor [shape=(..., M)]
        Complex roots.

    Returns
    -------
    Tensor [shape=(..., M+1)]
        Polynomial coefficients.

    """
    return nn.RootsToPolynomial._func(x)


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
    Tensor [shape=(...,)]
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
    Tensor [shape=(..., M+1)]
        Autocorrelation.

    """
    return nn.ReverseLevinsonDurbin._func(a)


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
    Tensor [shape=(..., M)]
        Roots of polynomial.

    """
    return nn.PolynomialToRoots._func(a, out_format=out_format)


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
    Tensor [shape=(..., L/2+1)]
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
    frame_period=160,
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
    Tensor [shape=(..., T/P, N/2+1)]
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
    Tensor [shape=(...,)]
        Compressed waveform.

    """
    return nn.MuLawCompression._func(x, abs_max=abs_max, mu=mu)


def unframe(
    y,
    *,
    out_length=None,
    frame_length=1,
    frame_period=1,
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

    center : bool [scalar]
        If True, assume that the center of data is the center of frame, otherwise
        assume that the center of data is the left edge of frame.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular']
        Window type.

    norm : ['none', 'power', 'magnitude']
        Normalization type of window.

    Returns
    -------
    Tensor [shape=(..., T)]
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


def window(x, *, out_length=None, window="blackman", norm="power"):
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
    Tensor [shape=(..., L2)]
        Windowed waveform.

    """
    return nn.Window._func(x, out_length=out_length, window=window, norm=norm)


def zcross(x, frame_length=1, norm=False):
    """Compute zero-crossing rate.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        Waveform.

    frame_length : int >= 1
        Frame length, :math:`L`.

    norm : bool
        If True, divide zero-crossing rate by frame length.

    Returns
    -------
    Tensor [shape=(..., T/L)]
        Zero-crossing rate.

    """
    return nn.ZeroCrossingAnalysis._func(x, frame_length=frame_length, norm=norm)
