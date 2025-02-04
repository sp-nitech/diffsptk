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

import importlib
from abc import ABC
from abc import abstractmethod

import torch
from torch import nn

from ..misc.utils import UNVOICED_SYMBOL
from ..misc.utils import numpy_to_torch
from .frame import Frame
from .stft import ShortTimeFourierTransform


class Pitch(nn.Module):
    """Pitch extraction module using external neural models.

    Parameters
    ----------
    frame_period : int >= 1
        Frame period, :math:`P`.

    sample_rate : int >= 1
        Sample rate in Hz.

    algorithm : ['crepe']
        Algorithm.

    out_format : ['pitch', 'f0', 'log-f0', 'prob', 'embed']
        Output format.

    f_min : float >= 0
        Minimum frequency in Hz.

    f_max : float <= sample_rate // 2
        Maximum frequency in Hz.

    voicing_threshold : float
        Voiced/unvoiced threshold.

    silence_threshold : float
        Silence threshold in dB.

    filter_length : int >= 1
        Window length of median and moving average filters.

    model : ['tiny', 'full']
        Model size.

    References
    ----------
    .. [1] J. W. Kim et al., "CREPE: A Convolutional Representation for Pitch
           Estimation," *Proceedings of ICASSP*, pp. 161-165, 2018.

    """

    def __init__(
        self,
        frame_period,
        sample_rate,
        algorithm="crepe",
        out_format="pitch",
        **kwargs,
    ):
        super().__init__()

        assert 1 <= frame_period
        assert 1 <= sample_rate

        if algorithm == "crepe":
            self.extractor = PitchExtractionByCrepe(frame_period, sample_rate, **kwargs)
        else:
            raise ValueError(f"algorithm {algorithm} is not supported.")

        def calc_pitch(x, convert, unvoiced_symbol=UNVOICED_SYMBOL):
            with torch.no_grad():
                y = self.extractor.calc_pitch(x)
                mask = y != UNVOICED_SYMBOL
                y[mask] = convert(y[mask])
                if unvoiced_symbol != UNVOICED_SYMBOL:
                    y[~mask] = unvoiced_symbol
            return y

        if out_format in (0, "pitch"):
            self.convert = lambda x: calc_pitch(x, lambda y: sample_rate / y)
        elif out_format in (1, "f0"):
            self.convert = lambda x: calc_pitch(x, lambda y: y)
        elif out_format in (2, "log-f0"):
            self.convert = lambda x: calc_pitch(x, lambda y: torch.log(y), -1e10)
        elif out_format == "prob":
            self.convert = lambda x: self.extractor.calc_prob(x)
        elif out_format == "embed":
            self.convert = lambda x: self.extractor.calc_embed(x)
        else:
            raise ValueError(f"out_format {out_format} is not supported.")

    def forward(self, x):
        """Compute pitch representation.

        Parameters
        ----------
        x : Tensor [shape=(B, T) or (T,)]
            Waveform.

        Returns
        -------
        out : Tensor [shape=(B, N, C) or (N, C) or (B, N) or (N,)]
            Pitch probability, embedding, or pitch, where N is the number of frames
            and C is the number of pitch classes or the dimension of embedding.

        Examples
        --------
        >>> x = diffsptk.sin(100, 10)
        >>> pitch = diffsptk.Pitch(80, 16000)
        >>> y = pitch(x)
        >>> y
        tensor([10.0860, 10.0860])

        """
        d = x.dim()
        if d == 1:
            x = x.unsqueeze(0)
        assert x.dim() == 2

        y = self.convert(x)

        if d == 1:
            y = y.squeeze(0)
        return y


class PitchExtractionInterface(ABC):
    """Abstract class for pitch extraction."""

    @abstractmethod
    def calc_prob(self, x):
        """Calculate pitch probability.

        Parameters
        ----------
        x : Tensor [shape=(B, T)]
            Waveform.

        Returns
        -------
        out : Tensor [shape=(B, N, C)]
            Probability, where C is the number of pitch classes.

        """

    @abstractmethod
    def calc_embed(self, x):
        """Calculate embedding.

        Parameters
        ----------
        x : Tensor [shape=(B, T)]
            Waveform.

        Returns
        -------
        out : Tensor [shape=(B, N, D)]
            Embedding, where D is the dimension of embedding.

        """

    @abstractmethod
    def calc_pitch(self, x):
        """Calculate pitch sequence.

        Parameters
        ----------
        x : Tensor [shape=(B, T)]
            Waveform.

        Returns
        -------
        out : Tensor [shape=(B, N)]
            F0 sequence.

        """


class PitchExtractionByCrepe(PitchExtractionInterface, nn.Module):
    """Pitch extraction by CREPE."""

    def __init__(
        self,
        frame_period,
        sample_rate,
        f_min=0,
        f_max=None,
        voicing_threshold=1e-2,
        silence_threshold=-60,
        filter_length=3,
        model="full",
    ):
        super().__init__()

        self.torchcrepe = importlib.import_module("torchcrepe")

        self.f_min = f_min
        self.f_max = self.torchcrepe.MAX_FMAX if f_max is None else f_max
        self.voicing_threshold = voicing_threshold
        self.silence_threshold = silence_threshold
        self.filter_length = filter_length
        self.model = model

        assert 0 <= self.f_min < self.f_max <= sample_rate / 2
        assert self.model in ("tiny", "full")

        if sample_rate != self.torchcrepe.SAMPLE_RATE:
            raise ValueError(
                f"Only {self.torchcrepe.SAMPLE_RATE} Hz is supported. "
                "Please use a resampler in advance."
            )

        self.frame = Frame(self.torchcrepe.WINDOW_SIZE, frame_period, zmean=True)
        self.stft = ShortTimeFourierTransform(
            self.torchcrepe.WINDOW_SIZE,
            frame_period,
            self.torchcrepe.WINDOW_SIZE,
            norm="none",
            window="hanning",
            out_format="db",
        )

        weights = self.torchcrepe.loudness.perceptual_weights().squeeze(-1)
        self.register_buffer("weights", numpy_to_torch(weights))

    def forward(self, x, embed=True):
        # torchcrepe.preprocess
        x = self.frame(x)
        x = x / torch.clip(x.std(dim=-1, keepdim=True), min=1e-10)

        # torchcrepe.infer
        B, N, L = x.shape
        x = x.reshape(-1, L)
        y = self.torchcrepe.infer(x, model=self.model, embed=embed, device=x.device)
        y = y.reshape(B, N, -1)
        return y

    def calc_prob(self, x):
        return self.forward(x, embed=False)

    def calc_embed(self, x):
        return self.forward(x, embed=True)

    def calc_pitch(self, x):
        # Compute pitch probabilities.
        prob = self.calc_prob(x).transpose(-2, -1)

        # Decode pitch probabilities.
        pitch, periodicity = self.torchcrepe.postprocess(
            prob,
            fmin=self.f_min,
            fmax=self.f_max,
            decoder=self.torchcrepe.decode.viterbi,
            return_harmonicity=False,
            return_periodicity=True,
        )

        # Apply filters.
        periodicity = self.torchcrepe.filter.median(periodicity, self.filter_length)
        org_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float)
        pitch = self.torchcrepe.filter.mean(pitch.float(), self.filter_length)
        torch.set_default_dtype(org_dtype)

        # Decide voiced/unvoiced.
        loudness = self.stft(x) + self.weights
        loudness = torch.clip(loudness, min=self.torchcrepe.loudness.MIN_DB)
        loudness = loudness.mean(-1)
        mask = torch.logical_or(
            periodicity < self.voicing_threshold, loudness < self.silence_threshold
        )
        pitch[mask] = UNVOICED_SYMBOL
        return pitch
