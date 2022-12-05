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

from abc import ABCMeta
from abc import abstractmethod

import torch
import torch.nn as nn
import torchcrepe

from ..misc.utils import default_dtype
from ..misc.utils import is_in
from .frame import Frame
from .stft import ShortTermFourierTransform

MAGIC_NUMBER_FOR_UNVOICED_FRAME = 0


class Pitch(nn.Module):
    """Pitch extraction module using external neural models.

    Parameters
    ----------
    frame_period : int >= 1 [scalar]
        Frame period, :math:`P`.

    sample_rate : int >= 1 [scalar]
        Sample rate in Hz.

    algorithm : ['crepe']
        Algorithm.

    out_format : ['pitch', 'f0', 'log-f0', 'prob', 'embed']
        Output format.

    f_min : float >= 0 [scalar]
        Minimum frequency in Hz.

    f_max : float <= sample_rate // 2 [scalar]
        Maximum frequency in Hz.

    threshold : float [scalar]
        Voiced/unvoiced threshold.

    silence_threshold : float [scalar]
        Silence threshold in dB.

    filter_length : int >= 1 [scalar]
        Window length of median and moving average filters.

    model : ['tiny', 'full']
        Model size.

    """

    def __init__(
        self,
        frame_period,
        sample_rate,
        algorithm="crepe",
        out_format="pitch",
        **kwargs,
    ):
        super(Pitch, self).__init__()

        assert 1 <= frame_period
        assert 1 <= sample_rate

        if algorithm == "crepe":
            self.extractor = PitchExtractionByCrepe(frame_period, sample_rate, **kwargs)
        else:
            raise ValueError(f"algorithm {algorithm} is not supported")

        def calc_pitch(x, convert, unvoiced_value):
            with torch.no_grad():
                y = self.extractor.calc_pitch(x)
                mask = y != MAGIC_NUMBER_FOR_UNVOICED_FRAME
                y[mask] = convert(y[mask])
                if unvoiced_value != MAGIC_NUMBER_FOR_UNVOICED_FRAME:
                    y[~mask] = unvoiced_value
            return y

        if out_format == 0 or out_format == "pitch":
            self.convert = lambda x: calc_pitch(x, lambda y: sample_rate / y, 0)
        elif out_format == 1 or out_format == "f0":
            self.convert = lambda x: calc_pitch(x, lambda y: y, 0)
        elif out_format == 2 or out_format == "log-f0":
            self.convert = lambda x: calc_pitch(x, lambda y: torch.log(y), -1e10)
        elif out_format == "prob":
            self.convert = lambda x: self.extractor.calc_prob(x)
        elif out_format == "embed":
            self.convert = lambda x: self.extractor.calc_embed(x)
        else:
            raise ValueError(f"out_format {out_format} is not supported")

    def forward(self, x):
        """Compute pitch representation.

        Parameters
        ----------
        x : Tensor [shape=(B, T) or (T,)]
            Waveform.

        Returns
        -------
        y : Tensor [shape=(B, N, C) or (N, C) or (B, N) or (N,)]
            Pitch probability, embedding, or pitch, where N is the number of frames
            and C is the number of pitch classes or the dimension of embedding.

        Examples
        --------
        >>> x = diffsptk.sin(100, 10)
        >>> pitch = diffsptk.Pitch(80, 16000)
        >>> y = pitch(x)
        >>> y
        tensor([1586.6013, 1593.9536])

        """
        d = x.dim()
        if d == 1:
            x = x.unsqueeze(0)
        assert x.dim() == 2

        y = self.convert(x)

        if d == 1:
            y = y.squeeze(0)
        return y


class PitchExtractionInterface(metaclass=ABCMeta):
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
        y : Tensor [shape=(B, N, C)]
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
        y : Tensor [shape=(B, N, D)]
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
        y : Tensor [shape=(B, N)]
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
        threshold=1e-2,
        silence_threshold=-60,
        filter_length=3,
        model="full",
    ):
        super(PitchExtractionByCrepe, self).__init__()

        self.f_min = f_min
        self.f_max = torchcrepe.MAX_FMAX if f_max is None else f_max
        self.threshold = threshold
        self.silence_threshold = silence_threshold
        self.filter_length = filter_length
        self.model = model

        assert 0 <= self.f_min < self.f_max <= sample_rate / 2
        assert is_in(self.model, ["tiny", "full"])

        if sample_rate != torchcrepe.SAMPLE_RATE:
            raise NotImplementedError(f"Only {torchcrepe.SAMPLE_RATE} Hz is supported")

        self.frame = Frame(torchcrepe.WINDOW_SIZE, frame_period, zmean=True)
        self.stft = ShortTermFourierTransform(
            torchcrepe.WINDOW_SIZE,
            frame_period,
            torchcrepe.WINDOW_SIZE,
            norm="none",
            window="hanning",
            out_format="db",
        )

        weights = torchcrepe.loudness.perceptual_weights().squeeze(-1)
        self.register_buffer(
            "weights", torch.from_numpy(weights.astype(default_dtype()))
        )

    def forward(self, x, embed=True):
        # torchcrepe.preprocess
        x = self.frame(x)
        x = x / torch.clip(x.std(dim=-1, keepdim=True), min=1e-10)

        # torchcrepe.infer
        B, N, L = x.shape
        x = x.reshape(-1, L)
        y = torchcrepe.infer(x, model=self.model, embed=embed)
        y = y.reshape(B, N, -1)
        return y

    def calc_prob(self, x):
        return self.forward(x, embed=False)

    def calc_embed(self, x):
        return self.forward(x, embed=True)

    def calc_pitch(self, x):
        # Compute pitch probabilities.
        prob = self.calc_prob(x).transpose(-1, -2)

        # Decode pitch probabilities.
        pitch, periodicity = torchcrepe.postprocess(
            prob,
            fmin=self.f_min,
            fmax=self.f_max,
            decoder=torchcrepe.decode.viterbi,
            return_harmonicity=False,
            return_periodicity=True,
        )

        # Apply filters.
        periodicity = torchcrepe.filter.median(periodicity, self.filter_length)
        pitch = torchcrepe.filter.mean(pitch, self.filter_length)

        # Decide voiced/unvoiced.
        loudness = self.stft(x) + self.weights
        loudness = torch.clip(loudness, min=torchcrepe.loudness.MIN_DB)
        loudness = loudness.mean(-1)
        mask = torch.logical_or(
            periodicity < self.threshold, loudness < self.silence_threshold
        )
        pitch[mask] = MAGIC_NUMBER_FOR_UNVOICED_FRAME
        return pitch
