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
import torchaudio
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

    algorithm : ['fcnf0', 'crepe']
        Algorithm.

    out_format : ['pitch', 'f0', 'log-f0', 'prob', 'embed']
        Output format.

    f_min : float >= 0
        Minimum frequency in Hz.

    f_max : float <= sample_rate // 2
        Maximum frequency in Hz.

    voicing_threshold : float
        Voiced/unvoiced threshold.

    References
    ----------
    .. [1] J. W. Kim et al., "CREPE: A Convolutional Representation for Pitch
           Estimation," *Proceedings of ICASSP*, pp. 161-165, 2018.

    .. [2] M. Morisson et al., "Cross-domain Neural Pitch and Periodicity Estimation,"
           *arXiv prepreint*, arXiv:2301.12258, 2023.

    """

    def __init__(
        self,
        frame_period,
        sample_rate,
        algorithm="fcnf0",
        out_format="pitch",
        **kwargs,
    ):
        super().__init__()

        assert 1 <= frame_period
        assert 1 <= sample_rate

        if algorithm == "fcnf0":
            self.extractor = PitchExtractionByFCNF0(frame_period, sample_rate, **kwargs)
        elif algorithm == "crepe":
            self.extractor = PitchExtractionByCREPE(frame_period, sample_rate, **kwargs)
        else:
            raise ValueError(f"algorithm {algorithm} is not supported.")

        @torch.inference_mode()
        def calc_pitch(x, convert, unvoiced_symbol=UNVOICED_SYMBOL):
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
        >>> x = diffsptk.sin(1000, 80)
        >>> pitch = diffsptk.Pitch(160, 8000, out_format="f0")
        >>> y = pitch(x)
        >>> y
        tensor([  0.0000,  99.7280,  99.7676,  99.8334,  99.8162, 100.1602,   0.0000])

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


class PitchExtractionByFCNF0(PitchExtractionInterface, nn.Module):
    """Pitch extraction by FCNF0."""

    def __init__(
        self,
        frame_period,
        sample_rate,
        f_min=None,
        f_max=None,
        voicing_threshold=0.5,
    ):
        super().__init__()

        try:
            self.penn = importlib.import_module("penn")
        except ImportError:
            raise ImportError("Please install torchcrepe by `pip install penn`.")

        self.f_min = self.penn.FMIN if f_min is None else f_min
        self.f_max = self.penn.FMAX if f_max is None else f_max
        assert 0 <= self.f_min < self.f_max <= sample_rate / 2

        self.voicing_threshold = voicing_threshold

        self.frame = Frame(
            self.penn.WINDOW_SIZE,
            frame_period * self.penn.SAMPLE_RATE // sample_rate,
            mode="reflect",
        )
        self.resample = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=self.penn.SAMPLE_RATE,
            dtype=torch.get_default_dtype(),
        )

    def forward(self, x):
        x = self.resample(x)
        frames = self.frame(x).reshape(-1, 1, self.penn.WINDOW_SIZE)
        logits = self.penn.infer(frames)
        logits = logits.reshape(x.size(0), -1, self.penn.PITCH_BINS)
        return logits

    def calc_prob(self, x):
        return torch.softmax(self.forward(x), dim=-1)

    def calc_embed(self, x):
        raise NotImplementedError

    def calc_pitch(self, x):
        logits = self.forward(x)
        logits = logits.reshape(-1, self.penn.PITCH_BINS, 1)
        result = self.penn.postprocess(logits)
        pitch = torch.where(
            self.voicing_threshold <= result[2], result[1], UNVOICED_SYMBOL
        )
        return pitch.reshape(x.size(0), -1)


class PitchExtractionByCREPE(PitchExtractionInterface, nn.Module):
    """Pitch extraction by CREPE."""

    def __init__(
        self,
        frame_period,
        sample_rate,
        f_min=None,
        f_max=None,
        voicing_threshold=1e-2,
        silence_threshold=-60,
        filter_length=3,
        model="full",
    ):
        super().__init__()

        try:
            self.torchcrepe = importlib.import_module("torchcrepe")
        except ImportError:
            raise ImportError("Please install torchcrepe by `pip install torchcrepe`.")

        self.f_min = 50 if f_min is None else f_min
        self.f_max = self.torchcrepe.MAX_FMAX if f_max is None else f_max
        assert 0 <= self.f_min < self.f_max <= sample_rate / 2

        self.voicing_threshold = voicing_threshold
        self.silence_threshold = silence_threshold
        self.filter_length = filter_length

        self.model = model
        assert self.model in ("tiny", "full")

        self.frame = Frame(
            self.torchcrepe.WINDOW_SIZE,
            frame_period * self.torchcrepe.SAMPLE_RATE // sample_rate,
            zmean=True,
        )
        self.stft = ShortTimeFourierTransform(
            self.torchcrepe.WINDOW_SIZE,
            frame_period * self.torchcrepe.SAMPLE_RATE // sample_rate,
            self.torchcrepe.WINDOW_SIZE,
            norm="none",
            window="hanning",
            out_format="db",
        )
        self.resample = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=self.torchcrepe.SAMPLE_RATE,
            dtype=torch.get_default_dtype(),
        )

        weights = self.torchcrepe.loudness.perceptual_weights().squeeze(-1)
        self.register_buffer("weights", numpy_to_torch(weights))

    def forward(self, x, embed=True):
        x = self.frame(x)
        x = x / torch.clip(x.std(dim=-1, keepdim=True), min=1e-10)

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
        prob = self.calc_prob(x).transpose(-2, -1)

        pitch, periodicity = self.torchcrepe.postprocess(
            prob,
            fmin=self.f_min,
            fmax=self.f_max,
            decoder=self.torchcrepe.decode.viterbi,
            return_harmonicity=False,
            return_periodicity=True,
        )

        periodicity = self.torchcrepe.filter.median(periodicity, self.filter_length)
        org_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float)
        pitch = self.torchcrepe.filter.mean(pitch.float(), self.filter_length)
        torch.set_default_dtype(org_dtype)

        loudness = self.stft(x) + self.weights
        loudness = torch.clip(loudness, min=self.torchcrepe.loudness.MIN_DB)
        loudness = loudness.mean(-1)
        mask = torch.logical_or(
            periodicity < self.voicing_threshold, loudness < self.silence_threshold
        )
        pitch[mask] = UNVOICED_SYMBOL
        return pitch
