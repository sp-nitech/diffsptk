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
from abc import ABC, abstractmethod

import torch
import torchaudio
from torch import nn

from ..utils.private import UNVOICED_SYMBOL, to
from .base import BaseNonFunctionalModule
from .frame import Frame
from .stft import ShortTimeFourierTransform


class Pitch(BaseNonFunctionalModule):
    """Pitch extraction module using external neural models.

    Parameters
    ----------
    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    sample_rate : int >= 8000
        The sample rate in Hz.

    algorithm : ['crepe', 'fcnf0']
        The algorithm to estimate pitch.

    out_format : ['pitch', 'f0', 'log-f0', 'prob', 'embed']
        The output format.

    f_min : float >= 0
        The minimum frequency in Hz.

    f_max : float <= sample_rate // 2
        The maximum frequency in Hz.

    voicing_threshold : float
        The voiced/unvoiced threshold.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] J. W. Kim et al., "CREPE: A convolutional representation for pitch
           estimation," *Proceedings of ICASSP*, pp. 161-165, 2018.

    .. [2] M. Morisson et al., "Cross-domain neural pitch and periodicity estimation,"
           *arXiv prepreint*, arXiv:2301.12258, 2023.

    """

    def __init__(
        self,
        frame_period: int,
        sample_rate: int,
        algorithm: str = "fcnf0",
        out_format: str | int = "pitch",
        **kwargs,
    ) -> None:
        super().__init__()

        if frame_period <= 0:
            raise ValueError("frame_period must be positive.")
        if sample_rate < 8000:
            raise ValueError("sample_rate must be at least 8000 Hz.")

        if algorithm == "crepe":
            self.extractor = PitchExtractionByCREPE(frame_period, sample_rate, **kwargs)
        elif algorithm == "fcnf0":
            self.extractor = PitchExtractionByFCNF0(frame_period, sample_rate, **kwargs)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pitch representation.

        Parameters
        ----------
        x : Tensor [shape=(B, T) or (T,)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(B, N, C) or (N, C) or (B, N) or (N,)]
            The pitch probability, embedding, or pitch, where N is the number of frames
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
        if x.dim() != 2:
            raise ValueError("Input must be 1D or 2D tensor.")

        y = self.convert(x)

        if d == 1:
            y = y.squeeze(0)
        return y


class PitchExtractionInterface(ABC, nn.Module):
    """Abstract class for pitch extraction."""

    @abstractmethod
    def calc_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the pitch probability.

        Parameters
        ----------
        x : Tensor [shape=(B, T)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(B, N, C)]
            The pitch probability, where C is the number of pitch classes.

        """
        raise NotImplementedError

    @abstractmethod
    def calc_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the pitch embedding.

        Parameters
        ----------
        x : Tensor [shape=(B, T)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(B, N, D)]
            The pitch embedding, where D is the dimension of embedding.

        """
        raise NotImplementedError

    @abstractmethod
    def calc_pitch(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the pitch sequence.

        Parameters
        ----------
        x : Tensor [shape=(B, T)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(B, N)]
            The pitch sequence.

        """
        raise NotImplementedError


class PitchExtractionByCREPE(PitchExtractionInterface):
    """Pitch extraction by CREPE."""

    def __init__(
        self,
        frame_period: int,
        sample_rate: int,
        *,
        f_min: float | None = None,
        f_max: float | None = None,
        voicing_threshold: float = 1e-2,
        silence_threshold: float = -60,
        filter_length: int = 3,
        model: str = "full",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        try:
            self.torchcrepe = importlib.import_module("torchcrepe")
        except ImportError:
            raise ImportError("Please install torchcrepe by `pip install torchcrepe`.")

        self.f_min = 50 if f_min is None else f_min
        self.f_max = self.torchcrepe.MAX_FMAX if f_max is None else f_max
        if not 0 <= self.f_min < self.f_max <= sample_rate / 2:
            raise ValueError("Invalid f_min and f_max.")

        self.voicing_threshold = voicing_threshold
        self.silence_threshold = silence_threshold
        self.filter_length = filter_length

        self.model = model
        if self.model not in ("tiny", "full"):
            raise ValueError("model must be 'tiny' or 'full'.")

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
            device=device,
            dtype=dtype,
        )
        self.resample = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=self.torchcrepe.SAMPLE_RATE,
            dtype=torch.get_default_dtype() if dtype is None else dtype,
        ).to(device)

        weights = self.torchcrepe.loudness.perceptual_weights().squeeze(-1)
        self.register_buffer("weights", to(weights, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, embed: bool = True) -> torch.Tensor:
        x = self.resample(x)
        x = self.frame(x)
        x = x / torch.clip(x.std(dim=-1, keepdim=True), min=1e-10)

        B, N, L = x.shape
        x = x.reshape(-1, L)
        y = self.torchcrepe.infer(
            x.float(), model=self.model, embed=embed, device=x.device
        )
        y = y.reshape(B, N, -1).to(dtype=x.dtype)
        return y

    def calc_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, embed=False)

    def calc_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, embed=True)

    def calc_pitch(self, x: torch.Tensor) -> torch.Tensor:
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
        pitch = self.torchcrepe.filter.mean(pitch.float(), self.filter_length).to(
            dtype=x.dtype
        )
        torch.set_default_dtype(org_dtype)

        loudness = self.stft(x) + self.weights
        loudness = torch.clip(loudness, min=self.torchcrepe.loudness.MIN_DB)
        loudness = loudness.mean(-1)
        mask = torch.logical_or(
            periodicity < self.voicing_threshold, loudness < self.silence_threshold
        )
        pitch[mask] = UNVOICED_SYMBOL
        return pitch


class PitchExtractionByFCNF0(PitchExtractionInterface):
    """Pitch extraction by FCNF0."""

    def __init__(
        self,
        frame_period: int,
        sample_rate: int,
        *,
        f_min: float | None = None,
        f_max: float | None = None,
        voicing_threshold: float = 0.5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        try:
            self.penn = importlib.import_module("penn")
        except ImportError:
            raise ImportError("Please install penn by `pip install penn`.")

        self.f_min = self.penn.FMIN if f_min is None else f_min
        self.f_max = self.penn.FMAX if f_max is None else f_max
        if not 0 <= self.f_min < self.f_max <= sample_rate / 2:
            raise ValueError("Invalid f_min and f_max.")

        self.voicing_threshold = voicing_threshold

        self.frame = Frame(
            self.penn.WINDOW_SIZE,
            frame_period * self.penn.SAMPLE_RATE // sample_rate,
            mode="reflect",
        )
        self.resample = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=self.penn.SAMPLE_RATE,
            dtype=torch.get_default_dtype() if dtype is None else dtype,
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resample(x)
        frames = self.frame(x)
        target_shape = frames.shape[:-1] + (self.penn.PITCH_BINS,)
        org_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float)
        logits = self.penn.infer(frames.float().reshape(-1, 1, self.penn.WINDOW_SIZE))
        torch.set_default_dtype(org_dtype)
        logits = logits.reshape(*target_shape).to(dtype=x.dtype)
        return logits

    def calc_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)

    def calc_embed(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    def calc_pitch(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        target_shape = logits.shape[:-1]
        logits = logits.reshape(-1, self.penn.PITCH_BINS, 1)
        result = self.penn.postprocess(logits)
        pitch = torch.where(
            self.voicing_threshold <= result[2], result[1], UNVOICED_SYMBOL
        )
        return pitch.reshape(*target_shape)
