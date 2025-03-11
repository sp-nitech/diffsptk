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

import numpy as np
import soundfile as sf
import torch


def get_alpha(sr, mode="hts", n_freq=10, n_alpha=100):
    """Compute an appropriate frequency warping factor given the sample rate.

    Parameters
    ----------
    sr : int >= 1
        The sample rate in Hz.

    mode : ['hts', 'auto']
        'hts' returns a traditional alpha used in HTS. 'auto' computes an appropriate
        alpha in the L2 sense.

    n_freq : int >= 2
        The number of sample points in the frequency domain.

    n_alpha : int >= 1
        The number of sample points to search alpha.

    Returns
    -------
    out : float in [0, 1)
        The frequency warping factor, :math:`\\alpha`.

    Examples
    --------
    >>> _, sr = diffsptk.read("assets/data.wav")
    >>> alpha = diffsptk.get_alpha(sr)
    >>> alpha
    0.42

    """

    def get_hts_alpha(sr):
        sr_to_alpha = {
            "8000": 0.31,
            "10000": 0.35,
            "12000": 0.37,
            "16000": 0.42,
            "22050": 0.45,
            "24000": 0.47,  # Added
            "32000": 0.50,
            "44100": 0.53,
            "48000": 0.55,
        }
        key = str(int(sr))
        if key not in sr_to_alpha:
            raise ValueError(f"Unsupported sample rate: {sr}. Please use mode='auto'.")
        selected_alpha = sr_to_alpha[key]
        return selected_alpha

    def get_auto_alpha(sr, n_freq, n_alpha):
        # Compute target mel-frequencies.
        freq = np.arange(n_freq) * (0.5 * sr / (n_freq - 1))
        mel_freq = np.log1p(freq / 1000)
        mel_freq = mel_freq * (np.pi / mel_freq[-1])
        mel_freq = np.expand_dims(mel_freq, 0)

        # Compute phase characteristic of the 1st order all-pass filter.
        alpha = np.linspace(0, 1, n_alpha, endpoint=False)
        alpha = np.expand_dims(alpha, 1)
        alpha2 = alpha * alpha
        omega = np.arange(n_freq) * (np.pi / (n_freq - 1))
        omega = np.expand_dims(omega, 0)
        numer = (1 - alpha2) * np.sin(omega)
        denom = (1 + alpha2) * np.cos(omega) - 2 * alpha
        warped_omega = np.arctan(numer / denom)
        warped_omega[warped_omega < 0] += np.pi

        # Select an appropriate alpha in terms of L2 distance.
        distance = np.square(mel_freq - warped_omega).sum(axis=1)
        selected_alpha = float(np.squeeze(alpha[np.argmin(distance)]))
        return selected_alpha

    if mode == "hts":
        alpha = get_hts_alpha(sr)
    elif mode == "auto":
        alpha = get_auto_alpha(sr, n_freq, n_alpha)
    else:
        raise ValueError("Only hts and auto are supported.")

    return alpha


def read(filename, device=None, dtype=None, **kwargs):
    """Read a waveform from the given file.

    Parameters
    ----------
    filename : str
        The path of the wav file.

    device : torch.device or None
        The device of the returned tensor.

    dtype : torch.dtype or None
        The data type of the returned tensor.

    **kwargs : additional keyword arguments
        Additional arguments passed to `soundfile.read
        <https://python-soundfile.readthedocs.io/en/latest/#soundfile.read>`_.

    Returns
    -------
    x : Tensor
        The waveform.

    sr : int
        The sample rate in Hz.

    Examples
    --------
    >>> x, sr = diffsptk.read("assets/data.wav")
    >>> x
    tensor([ 0.0002,  0.0004,  0.0006,  ...,  0.0006, -0.0006, -0.0007])
    >>> sr
    16000

    """
    x, sr = sf.read(filename, **kwargs)
    if dtype is None:
        dtype = torch.get_default_dtype()
    x = torch.tensor(x, device=device, dtype=dtype)
    return x, sr


def write(filename, x, sr, **kwargs):
    """Write the given waveform to a file.

    Parameters
    ----------
    filename : str
        The path of the wav file.

    x : Tensor
        The waveform.

    sr : int
        The sample rate in Hz.

    **kwargs : additional keyword arguments
        Additional arguments passed to `soundfile.write
        <https://python-soundfile.readthedocs.io/en/latest/#soundfile.write>`_.

    Examples
    --------
    >>> x, sr = diffsptk.read("assets/data.wav")
    >>> diffsptk.write("out.wav", x, sr)

    """
    x = x.cpu().numpy() if torch.is_tensor(x) else x
    sf.write(filename, x, sr, **kwargs)
