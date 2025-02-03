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

import librosa
import pytest
import torch

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
def test_compatibility(device, module, C=12):
    if device == "cuda" and not torch.cuda.is_available():
        return

    x, sr = diffsptk.read(
        "assets/data.wav",
        double=torch.get_default_dtype() == torch.double,
        device=device,
    )

    X = diffsptk.functional.stft(x)

    c1 = librosa.feature.chroma_stft(
        S=X.cpu().numpy().T,
        sr=sr,
        n_chroma=C,
        tuning=0,
    ).T

    chroma = U.choice(
        module,
        diffsptk.ChromaFilterBankAnalysis,
        diffsptk.functional.chroma,
        {"fft_length": 2 * (X.size(-1) - 1)},
        {"n_channel": C, "sample_rate": sr},
    )
    if hasattr(chroma, "to"):
        chroma.to(device)
    c2 = chroma(X).cpu().numpy()

    assert U.allclose(c1, c2)

    U.check_differentiability(device, chroma, [X.size(-1)])
