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
    return nn.MLSADigitalFilterCoefficientsToMelCepstrum._forward(b, alpha=alpha)


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
    return nn.CepstrumToNegativeDerivativeOfPhaseSpectrum._forward(
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
    return nn.Decimation._forward(x, period=period, start=start, dim=dim)


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
    return nn.Delay._forward(x, start=start, keeplen=keeplen, dim=dim)


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
    return nn.InverseUniformQuantization._forward(
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
    return nn.ExcitationGeneration._forward(
        p,
        frame_period=frame_period,
        voiced_region=voiced_region,
        unvoiced_region=unvoiced_region,
    )


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
    return nn.Frame._forward(
        x,
        frame_length=frame_length,
        frame_period=frame_period,
        center=center,
        zmean=zmean,
    )


def gnorm(x, gamma=0):
    """Perform cepstrum gain normalization.

    Parameters
    ----------
    x : Tensor [shape=(..., M+1)]
        Generalized cepstrum.

    Parameters
    ----------
    gamma : float in [-1, 1]
        Gamma, :math:`\\gamma`.

    Returns
    -------
    Tensor [shape=(..., M+1)]
        Normalized generalized cepstrum.

    """
    return nn.GeneralizedCepstrumGainNormalization._forward(x, gamma=gamma)


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
    return nn.GroupDelay._forward(
        b, a, fft_length=fft_length, alpha=alpha, gamma=gamma, **kwargs
    )


def ignorm(y, gamma=0):
    """Perform cepstrum inverse gain normalization.

    Parameters
    ----------
    y : Tensor [shape=(..., M+1)]
        Normalized generalized cepstrum.

    gamma : float in [-1, 1]
        Gamma, :math:`\\gamma`.

    Returns
    -------
    x : Tensor [shape=(..., M+1)]
        Generalized cepstrum.

    """
    return nn.GeneralizedCepstrumInverseGainNormalization._forward(y, gamma=gamma)


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
    return nn.Interpolation._forward(x, period=period, start=start, dim=dim)


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
    return nn.LinearInterpolation._forward(x, upsampling_factor=upsampling_factor)


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
    return nn.MagicNumberInterpolation._forward(x, magic_number=magic_number)


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
    return nn.MelCepstrumToMLSADigitalFilterCoefficients._forward(mc, alpha=alpha)


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
    return nn.NegativeDerivativeOfPhaseSpectrumToCepstrum._forward(
        n, cep_order=cep_order
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
    Tensor [shape=(..., L/2+1)]
        Phase spectrum [:math:`\\pi` rad].

    """
    return nn.Phase._forward(b, a, fft_length=fft_length, unwrap=unwrap)


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
    return nn.UniformQuantization._forward(
        x, abs_max=abs_max, n_bit=n_bit, quantizer=quantizer
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
    Tensor [shape=(..., L/2+1)]
        Spectrum.

    """
    return nn.Spectrum._forward(
        b,
        a,
        fft_length=fft_length,
        eps=eps,
        relative_floor=relative_floor,
        out_format=out_format,
    )


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
    return nn.ZeroCrossingAnalysis._forward(x, frame_length=frame_length, norm=norm)
