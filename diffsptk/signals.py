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

import math

import torch


def impulse(order: int, **kwargs) -> torch.Tensor:
    """Generate impulse sequence.

    See `impulse <https://sp-nitech.github.io/sptk/latest/main/impulse.html>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        The order of the sequence, :math:`M`.

    **kwargs : additional keyword arguments
        See `torch.eye <https://pytorch.org/docs/stable/generated/torch.eye.html>`_.

    Returns
    -------
    out : Tensor [shape=(M+1,)]
        The impulse sequence.

    Examples
    --------
    >>> import diffsptk
    >>> x = diffsptk.impulse(4)
    >>> x
    tensor([1., 0., 0., 0., 0.])

    """
    x = torch.eye(n=1, m=order + 1, **kwargs).squeeze(0)
    return x


def step(order: int, value: float = 1, **kwargs) -> torch.Tensor:
    """Generate step sequence.

    See `step <https://sp-nitech.github.io/sptk/latest/main/step.html>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        The order of the sequence, :math:`M`.

    value : float
        Step value.

    **kwargs : additional keyword arguments
        See `torch.full <https://pytorch.org/docs/stable/generated/torch.full.html>`_.

    Returns
    -------
    out : Tensor [shape=(M+1,)]
        The step sequence.

    Examples
    --------
    >>> import diffsptk
    >>> x = diffsptk.step(4, 2)
    >>> x
    tensor([2., 2., 2., 2., 2.])

    """
    x = torch.full((order + 1,), float(value), **kwargs)
    return x


def ramp(
    arg: float, end: float | None = None, step: float = 1, eps: float = 1e-8, **kwargs
) -> torch.Tensor:
    """Generate ramp sequence.

    See `ramp <https://sp-nitech.github.io/sptk/latest/main/ramp.html>`_
    for details.

    Parameters
    ----------
    arg : float
        If `end` is `None`, this is the end value otherwise start value.

    end : float or None
        The end value.

    step : float != 0
        The slope.

    eps : float
        A correction value.

    **kwargs : additional keyword arguments
        See `torch.arange
        <https://pytorch.org/docs/stable/generated/torch.arange.html>`_.

    Returns
    -------
    out : Tensor [shape=(?,)]
        The ramp sequence.

    Examples
    --------
    >>> import diffsptk
    >>> x = diffsptk.ramp(4)
    >>> x
    tensor([0., 1., 2., 3., 4.])

    """
    if end is None:
        start = 0
        end = arg
    else:
        start = arg
    if 0 < step:
        end += eps
    elif step < 0:
        end -= eps
    else:
        raise ValueError("step must be non-zero")
    x = torch.arange(start, end, step, **kwargs)
    return x


def sin(
    order: int, period: float | None = None, magnitude: float = 1, **kwargs
) -> torch.Tensor:
    """Generate sinusoidal sequence.

    See `sin <https://sp-nitech.github.io/sptk/latest/main/sin.html>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        The order of the sequence, :math:`M`.

    period : float > 0
        The period.

    magnitude : float
        The magnitude.

    **kwargs : additional keyword arguments
        See `torch.arange
        <https://pytorch.org/docs/stable/generated/torch.arange.html>`_.

    Returns
    -------
    out : Tensor [shape=(M+1,)]
        The sinusoidal sequence.

    Examples
    --------
    >>> import diffsptk
    >>> x = diffsptk.sin(4)
    >>> x
    tensor([ 0.0000,  0.9511,  0.5878, -0.5878, -0.9511])

    """
    if period is None:
        period = order + 1
    if period <= 0:
        raise ValueError("period must be positive.")
    x = torch.arange(order + 1, **kwargs)
    x = torch.sin(x * (2 * torch.pi / period)) * magnitude
    return x


def train(
    order: int, frame_period: float, norm: str | int = "power", **kwargs
) -> torch.Tensor:
    """Generate pulse sequence.

    See `train <https://sp-nitech.github.io/sptk/latest/main/train.html>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        The order of the sequence, :math:`M`.

    frame_period : float >= 1
        The frame period.

    norm : ['none', 'power', 'magnitude']
        The normalization type.

    **kwargs : additional keyword arguments
        See `torch.zeros <https://pytorch.org/docs/stable/generated/torch.zeros.html>`_.

    Returns
    -------
    out : Tensor [shape=(M+1,)]
        The pulse sequence.

    Examples
    --------
    >>> import diffsptk
    >>> x = diffsptk.train(5, 2.3)
    >>> x
    tensor([1.5166, 0.0000, 0.0000, 1.5166, 0.0000, 1.5166])

    """
    if frame_period < 1:
        raise ValueError("frame_period must be greater than or equal to 1.")

    if norm in (0, "none"):
        pulse = 1
    elif norm in (1, "power"):
        pulse = frame_period**0.5
    elif norm in (2, "magnitude"):
        pulse = frame_period
    else:
        raise ValueError(f"norm {norm} is not supported.")

    frequency = 1 / frame_period
    v = torch.full((order + 2,), frequency)
    v[0] *= -1
    v = torch.floor(torch.cumsum(v, dim=0))
    index = torch.ge(torch.diff(v), 1)

    x = torch.zeros(order + 1, **kwargs)
    x[index] = pulse
    return x


def mseq(*order: int, **kwargs) -> torch.Tensor:
    """Generate M-sequence.

    See `mseq <https://sp-nitech.github.io/sptk/latest/main/mseq.html>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        The order of the sequence, :math:`M`.

    **kwargs : additional keyword arguments
        See `torch.ones <https://pytorch.org/docs/stable/generated/torch.ones.html>`_.

    Returns
    -------
    out : Tensor [shape=(M+1,)]
        The random value sequence.

    Examples
    --------
    >>> import diffsptk
    >>> x = diffsptk.mseq(4)
    >>> x
    tensor([-1.,  1., -1.,  1., -1.])

    """
    if any(isinstance(item, (list, tuple)) for item in order):
        order = list(*order)
    else:
        order = list(order)
    order[-1] += 1

    out_length = math.prod(order)
    out = torch.ones(out_length, **kwargs)

    init = 0x55555555  # int("01010101010101010101010101010101", 2)
    B0 = 0x00000001  # 1 << 0
    B28 = 0x10000000  # 1 << 28
    B31 = 0x80000000  # 1 << 31
    B31F = 0x7FFFFFFF  # ~(1 << 31) & 0xFFFFFFFF

    x = init
    for i in range(out_length):
        x >>= 1

        x0 = 1 if x & B0 else -1
        x28 = 1 if x & B28 else -1

        if x0 + x28:
            x &= B31F
        else:
            x |= B31

        if x0 != 1:
            out[i] = x0

    return out.reshape(order)


def nrand(
    *order: int, mean: float = 0, stdv: float = 1, var: float | None = None, **kwargs
) -> torch.Tensor:
    """Generate Gaussian random number sequence.

    See `nrand <https://sp-nitech.github.io/sptk/latest/main/nrand.html>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        The order of the sequence, :math:`M`.

    mean : float
        The mean.

    stdv : float >= 0
        The standard deviation.

    var : float >= 0
        The variance. This overrides `stdv`.

    **kwargs : additional keyword arguments
        See `torch.randn <https://pytorch.org/docs/stable/generated/torch.randn.html>`_.

    Returns
    -------
    out : Tensor [shape=(M+1,)]
        The random value sequence.

    Examples
    --------
    >>> import diffsptk
    >>> x = diffsptk.nrand(4)
    >>> x.shape
    torch.Size([5])
    >>> x = diffsptk.nrand(2, 4)
    >>> x.shape
    torch.Size([2, 5])

    """
    if var is not None:
        stdv = var**0.5
    if stdv < 0:
        raise ValueError("stdv must be non-negative.")

    if any(isinstance(item, (list, tuple)) for item in order):
        order = list(*order)
    else:
        order = list(order)
    order[-1] += 1
    x = torch.randn(*order, **kwargs)
    x = x * stdv + mean
    return x


def rand(*order: int, a: float = 0, b: float = 1, **kwargs) -> torch.Tensor:
    """Generate uniform random number sequence.

    See `rand <https://sp-nitech.github.io/sptk/latest/main/rand.html>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        The order of the sequence, :math:`M`.

    a : float
        The lower bound.

    b : float >= 0
        The upper bound.

    **kwargs : additional keyword arguments
        See `torch.rand <https://pytorch.org/docs/stable/generated/torch.rand.html>`_.

    Returns
    -------
    out : Tensor [shape=(M+1,)]
        The random value sequence.

    Examples
    --------
    >>> import diffsptk
    >>> x = diffsptk.rand(4)
    >>> x.shape
    torch.Size([5])
    >>> x = diffsptk.rand(2, 4)
    >>> x.shape
    torch.Size([2, 5])

    """
    if b <= a:
        raise ValueError("Lower bound must be less than upper bound.")

    if any(isinstance(item, (list, tuple)) for item in order):
        order = list(*order)
    else:
        order = list(order)
    order[-1] += 1
    x = torch.rand(*order, **kwargs)
    x = (b - a) * x + a
    return x
