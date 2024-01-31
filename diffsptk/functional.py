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

from . import modules as fm


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

    Examples
    --------
    >>> x = diffsptk.ramp(9)
    >>> y = diffsptk.functional.decimate(x, 3, start=1)
    >>> y
    tensor([1., 4., 7.])

    """
    return fm.Decimation._forward(x, period=period, start=start, dim=dim)


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

    Examples
    --------
    >>> x = diffsptk.ramp(1, 3)
    >>> y = diffsptk.functional.delay(x, 2)
    >>> y
    tensor([0., 0., 1., 2., 3.])
    >>> y = diffsptk.functional.delay(x, 2, keeplen=True)
    >>> y
    tensor([0., 0., 1.])

    """
    return fm.Delay._forward(x, start=start, keeplen=keeplen, dim=dim)


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

    Examples
    --------
    >>> x = diffsptk.ramp(3)
    >>> g = diffsptk.functional.grpdelay(x, fft_length=8)
    >>> g
    tensor([2.3333, 2.4278, 3.0000, 3.9252, 3.0000])

    """
    return fm.GroupDelay._forward(
        b, a, fft_length=fft_length, alpha=alpha, gamma=gamma, **kwargs
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
    Tensor [shape=(..., TxP+S, ...)]
        Interpolated signal.

    Examples
    --------
    >>> x = diffsptk.ramp(1, 3)
    >>> y = diffsptk.functional.interpolate(x, 3, start=1)
    >>> y
    tensor([0., 1., 0., 0., 2., 0., 0., 3., 0., 0.])

    """
    return fm.Interpolation._forward(x, period=period, start=start, dim=dim)


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

    Examples
    --------
    >>> x = diffsptk.ramp(3)
    >>> p = diffsptk.functional.phase(x, fft_length=8)
    >>> p
    tensor([ 0.0000, -0.5907,  0.7500, -0.1687,  1.0000])

    """
    return fm.Phase._forward(b, a, fft_length=fft_length, unwrap=unwrap)


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

    Examples
    --------
    >>> x = diffsptk.ramp(1, 3)
    tensor([1., 2., 3.])
    >>> s = diffsptk.functional.spec(x, fft_length=8)
    >>> s
    tensor([36.0000, 25.3137,  8.0000,  2.6863,  4.0000])

    """
    return fm.Spectrum._forward(
        b,
        a,
        fft_length=fft_length,
        eps=eps,
        relative_floor=relative_floor,
        out_format=out_format,
    )
