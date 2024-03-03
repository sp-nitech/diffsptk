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


def impulse(order, **kwargs):
    """Generate impulse sequence.

    See `impulse <https://sp-nitech.github.io/sptk/latest/main/impulse.html>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        Order of sequence, :math:`M`.

    **kwargs : additional keyword arguments
        See `torch.eye <https://pytorch.org/docs/stable/generated/torch.eye.html>`_.

    Returns
    -------
    out : Tensor [shape=(M+1,)]
        Impulse sequence.

    Examples
    --------
    >>> x = diffsptk.impulse(4)
    >>> x
    tensor([1., 0., 0., 0., 0.])

    """
    x = torch.eye(n=1, m=order + 1, **kwargs).squeeze(0)
    return x


def step(order, value=1, **kwargs):
    """Generate step sequence.

    See `step <https://sp-nitech.github.io/sptk/latest/main/step.html>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        Order of sequence, :math:`M`.

    value : float
        Step value.

    **kwargs : additional keyword arguments
        See `torch.full <https://pytorch.org/docs/stable/generated/torch.full.html>`_.

    Returns
    -------
    out : Tensor [shape=(M+1,)]
        Step sequence.

    Examples
    --------
    >>> x = diffsptk.step(4, 2)
    >>> x
    tensor([2., 2., 2., 2., 2.])

    """
    x = torch.full((order + 1,), float(value), **kwargs)
    return x


def ramp(arg, end=None, step=1, eps=1e-8, **kwargs):
    """Generate ramp sequence.

    See `ramp <https://sp-nitech.github.io/sptk/latest/main/ramp.html>`_
    for details.

    Parameters
    ----------
    arg : float
        If `end` is `None` end value otherwise start value.

    end : float
        End value.

    step : float != 0
        Slope.

    eps : float
        A correction value.

    **kwargs : additional keyword arguments
        See `torch.arange
        <https://pytorch.org/docs/stable/generated/torch.arange.html>`_.

    Returns
    -------
    out : Tensor [shape=(?,)]
        Ramp sequence.

    Examples
    --------
    >>> x = diffsptk.ramp(4)
    >>> x
    tensor([0., 1., 2., 3., 4.])

    """
    if end is None:
        start = 0
        end = arg
    else:
        start = arg
    if step > 0:
        end += eps
    else:
        end -= eps
    x = torch.arange(start, end, step, **kwargs)
    return x


def sin(order, period=None, magnitude=1, **kwargs):
    """Generate sinusoidal sequence.

    See `sin <https://sp-nitech.github.io/sptk/latest/main/sin.html>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        Order of sequence, :math:`M`.

    period : float > 0
        Period.

    magnitude : float
        Magnitude.

    **kwargs : additional keyword arguments
        See `torch.arange
        <https://pytorch.org/docs/stable/generated/torch.arange.html>`_.

    Returns
    -------
    out : Tensor [shape=(M+1,)]
        Sinusoidal sequence.

    Examples
    --------
    >>> x = diffsptk.sin(4)
    >>> x
    tensor([ 0.0000,  0.9511,  0.5878, -0.5878, -0.9511])

    """
    if period is None:
        period = order + 1
    x = torch.arange(order + 1, **kwargs)
    x = torch.sin(x * (2 * torch.pi / period)) * magnitude
    return x


def train(order, frame_period, norm="power", **kwargs):
    """Generate pulse sequence.

    See `train <https://sp-nitech.github.io/sptk/latest/main/train.html>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        Order of sequence, :math:`M`.

    frame_period : float >= 1
        Frame period.

    norm : ['none', 'power', 'magnitude']
        Normalization type.

    **kwargs : additional keyword arguments
        See `torch.zeros <https://pytorch.org/docs/stable/generated/torch.zeros.html>`_.

    Returns
    -------
    out : Tensor [shape=(M+1,)]
        Pulse sequence.

    Examples
    --------
    >>> x = diffsptk.train(5, 2.3)
    >>> x
    tensor([1.5166, 0.0000, 0.0000, 1.5166, 0.0000, 1.5166])

    """
    assert 1 <= frame_period

    if norm == 0 or norm == "none":
        pulse = 1
    elif norm == 1 or norm == "power":
        pulse = frame_period**0.5
    elif norm == 2 or norm == "magnitude":
        pulse = frame_period
    else:
        raise ValueError(f"norm {norm} is not supported.")

    frequency = 1 / frame_period
    v = torch.full((order + 2,), frequency)
    v[0] *= -1
    v = torch.floor(torch.cumsum(v, dim=0))
    index = torch.ge(v[..., 1:] - v[..., :-1], 1)

    x = torch.zeros(order + 1, **kwargs)
    x[index] = pulse
    return x


def nrand(*order, mean=0, stdv=1, var=None, **kwargs):
    """Generate random number sequence.

    See `nrand <https://sp-nitech.github.io/sptk/latest/main/nrand.html>`_
    for details.

    Parameters
    ----------
    order : int >= 0
        Order of sequence, :math:`M`.

    mean : float
        Mean.

    stdv : float >= 0
        Standard deviation.

    var : float >= 0
        Variance.

    **kwargs : additional keyword arguments
        See `torch.randn <https://pytorch.org/docs/stable/generated/torch.randn.html>`_.

    Returns
    -------
    out : Tensor [shape=(M+1,)]
        Random value sequence.

    Examples
    --------
    >>> x = diffsptk.nrand(4)
    >>> x
    tensor([-0.8603,  0.6743, -0.9178,  1.5382, -0.2574])
    >>> x = diffsptk.nrand(2, 4)
    >>> x
    tensor([[-0.2385, -0.0778, -0.0418, -1.6217,  0.1560],
            [ 1.6646,  0.8429,  0.9357, -0.5123,  0.9571]])

    """
    if var is not None:
        stdv = var**0.5
    assert 0 <= stdv

    order = list(order)
    order[-1] += 1
    x = torch.randn(*order, **kwargs)
    x = x * stdv + mean
    return x
