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
import torch.nn as nn
import torchcrepe

from .frame import Frame


class Pitch(nn.Module):
    """Pitch extraction module using external neural models.

    Parameters
    ----------
    frame_period : int >= 1 [scalar]
        Frame period, :math:`P`.

    sample_rate : int >= 1 [scalar]
        Sample rate in Hz.

    f_min : float >= 0 [scalar]
        Minimum frequency in Hz.

    f_max : float <= sample_rate // 2 [scalar]
        Maximum frequency in Hz.

    algorithm : ['crepe']
        Algorithm.

    option : str -> Any [dict]
        Algorithm-dependent options.

    """

    def __init__(
        self,
        frame_period,
        sample_rate,
        f_min=0,
        f_max=None,
        algorithm="crepe",
        out_format="f0",
        **option,
    ):
        super(Pitch, self).__init__()

        assert 1 <= frame_period
        assert 1 <= sample_rate
        assert 0 <= f_min
        if f_max is not None:
            assert f_min < f_max
            assert f_max <= sample_rate / 2

        if algorithm == "crepe":
            self.algorithm = PitchExtractionByCrepe(
                frame_period,
                sample_rate,
                f_min=f_min,
                f_max=f_max,
                **option,
            )
        else:
            raise ValueError(f"algorithm {algorithm} is not supported")

        if out_format == 0 or out_format == "pitch":
            self.convert = lambda x: sample_rate / x
        elif out_format == 1 or out_format == "f0":
            self.convert = lambda x: x
        elif out_format == 2 or out_format == "lf0":
            self.convert = lambda x: torch.log(x)
        else:
            raise ValueError(f"out_format {out_format} is not supported")

    def forward(self, x, embed=False):
        """Compute pitch representation.

        Parameters
        ----------
        x : Tensor [shape=(B, T) or (T,)]
            Waveform.

        embed : bool [scalar]
            If True, return embedding instead of probability.

        Returns
        -------
        y : Tensor [shape=(B, N, C) or (N, C)]
            Pitch probability or embedding, where N is the number of frames and
            C is the number of classes or the dimension of embedding.

        Examples
        --------
        >>> x = diffsptk.sin(100, 10)
        >>> pitch = diffsptk.pitch(80, 16000)
        >>> prob = pitch(x)
        >>> prob.shape
        torch.Size([2, 360])

        """
        d = x.dim()
        if d == 1:
            x = x.unsqueeze(0)
        assert x.dim() == 2
        y = self.algorithm.forward(x, embed=embed)
        if d == 1:
            y = y.squeeze(0)
        return y

    def decode(self, prob):
        """Get appropriate pitch contour from pitch probabilities.

        Parameters
        ----------
        prob : Tensor [shape=(B, N, C) or (N, C)]
            Pitch probabilitiy.

        Returns
        -------
        pitch : Tensor [shape=(B, N) or (N,)]
            Pitch in seconds, Hz, or log Hz.

        Examples
        --------
        >>> x = diffsptk.sin(100, 10)
        >>> pitch = diffsptk.pitch(80, 16000)
        >>> prob = pitch.forward(x)
        >>> result = pitch.decode(prob)
        >>> result
        tensor([1586.6013, 1593.9536])

        """
        with torch.no_grad():
            d = prob.dim()
            if d == 2:
                prob = prob.unsqueeze(0)
            assert prob.dim() == 3
            pitch = self.algorithm.decode(prob)
            if d == 2:
                pitch = pitch.squeeze(0)
            pitch = self.convert(pitch)
        return pitch


class PitchExtractionByCrepe(nn.Module):
    def __init__(self, frame_period, sample_rate, f_min=0, f_max=None, model="full"):
        super(PitchExtractionByCrepe, self).__init__()

        self.f_min = f_min
        self.f_max = torchcrepe.MAX_FMAX if f_max is None else f_max
        self.model = model

        if sample_rate != torchcrepe.SAMPLE_RATE:
            raise NotImplementedError(f"Only {torchcrepe.SAMPLE_RATE} Hz is supported")

        self.frame = Frame(torchcrepe.WINDOW_SIZE, frame_period, zmean=True)

    def forward(self, x, embed=False):
        # torchcrepe.preprocess
        x = self.frame(x)
        x = x / torch.clip(x.std(dim=-1, keepdim=True), min=1e-10)

        # torchcrepe.infer
        B, N, L = x.shape
        x = x.reshape(-1, L)
        y = torchcrepe.infer(x, model=self.model, embed=embed)
        y = y.reshape(B, N, -1)
        return y

    def decode(self, prob):
        prob = prob.transpose(-1, -2)
        pitch = torchcrepe.postprocess(prob, fmin=self.f_min, fmax=self.f_max)
        return pitch
