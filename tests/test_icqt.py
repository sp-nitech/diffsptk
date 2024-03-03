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
import numpy as np
import pytest
import torch

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fp", [127, 128])
@pytest.mark.parametrize("scale", [False, True])
def test_compatibility(device, fp, scale, K=252, B=36, f_min=32.7, verbose=False):
    if device == "cuda" and not torch.cuda.is_available():
        return

    x, sr = diffsptk.read(librosa.ex("trumpet"))
    T = x.size(0)

    if verbose:
        diffsptk.write("original.wav", x, sr)

    c1 = librosa.cqt(
        x.cpu().numpy(),
        sr=sr,
        fmin=f_min,
        n_bins=K,
        bins_per_octave=B,
        hop_length=fp,
        scale=scale,
        res_type="kaiser_best",
        dtype=None,
    )

    y1 = librosa.icqt(
        c1,
        sr=sr,
        fmin=f_min,
        bins_per_octave=B,
        hop_length=fp,
        scale=scale,
        res_type="kaiser_best",
        dtype=np.float64 if torch.get_default_dtype() == torch.double else np.float32,
        length=T,
    )

    if verbose:
        diffsptk.write("liborsa.wav", y1, sr)

    icqt = diffsptk.ICQT(
        fp, sr, f_min=f_min, n_bin=K, n_bin_per_octave=B, scale=scale
    ).to(device)
    c2 = torch.from_numpy(c1).to(device).T
    y2 = icqt(c2, out_length=T).cpu().numpy()

    if verbose:
        diffsptk.write("diffsptk.wav", y2, sr)

    error = np.mean(np.abs(y1 - y2))
    assert error < 1e-4, f"Mean error: {error}"

    U.check_differentiability(device, icqt, [1, K], dtype=c2.dtype)
