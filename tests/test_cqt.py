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

import numpy as np
import pytest
import torch

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("fp", [511, 512])
@pytest.mark.parametrize("K", [1, 24])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("res_type", ["kaiser_best", "kaiser_fast"])
def test_compatibility(device, dtype, fp, K, scale, res_type, B=12, f_min=32.7):
    x, sr = diffsptk.read("assets/data.wav", device=device, dtype=dtype)

    cqt = diffsptk.CQT(
        fp,
        sr,
        f_min=f_min,
        n_bin=K,
        n_bin_per_octave=B,
        scale=scale,
        res_type=res_type,
        device=device,
        dtype=dtype,
    )

    try:  # pragma: no cover
        librosa = importlib.import_module("librosa")

        c1 = librosa.cqt(
            x.cpu().numpy(),
            sr=sr,
            fmin=f_min,
            n_bins=K,
            bins_per_octave=B,
            hop_length=fp,
            scale=scale,
            res_type=res_type,
            dtype=None,
        ).T
        c2 = cqt(x).cpu().numpy()
        c1 = c1[: c2.shape[0]]
        assert np.corrcoef(c1.real.flatten(), c2.real.flatten())[0, 1] > 0.99
        assert np.corrcoef(c1.imag.flatten(), c2.imag.flatten())[0, 1] > 0.99
    except ImportError:
        pass

    U.check_differentiability(device, dtype, [torch.abs, cqt], [fp])
