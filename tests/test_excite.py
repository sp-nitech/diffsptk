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
import pytest
import soundfile as sf
import torch

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("voiced_region", ["pulse", "sinusoidal"])
@pytest.mark.parametrize("unvoiced_region", ["gauss", "zeros"])
def test_compatibility(device, module, voiced_region, unvoiced_region, P=80):
    if device == "cuda" and not torch.cuda.is_available():
        return

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    excite = U.choice(
        module,
        diffsptk.ExcitationGeneration,
        diffsptk.functional.excite,
        {},
        {
            "frame_period": P,
            "voiced_region": voiced_region,
            "unvoiced_region": unvoiced_region,
        },
    )

    # Compute pitch and excitation on C++ version.
    cmd = "x2x +sd tools/SPTK/asset/data.short | "
    cmd += f"pitch -s 16 -p {P} -o 0 -a 2 > excite.tmp1"
    n = 0 if unvoiced_region == "zeros" else 1
    U.call(cmd, get=False)
    U.call(f"excite -p {P} -n {n} excite.tmp1 > excite.tmp2", get=False)

    # Compute excitation on PyTorch version.
    pitch = U.call("cat excite.tmp1")
    pitch = np.expand_dims(pitch, 0)  # This is to cover a case.
    e = excite(torch.from_numpy(pitch).to(device))
    e.cpu().double().numpy().tofile("excite.tmp3")

    def compute_error(infile):
        cmd = f"sopr -magic 0 -m 10 -MAGIC 0 {infile} | "
        cmd += f"pitch -s 16 -p {P} -o 0 -a 2"
        pitch = U.call("cat excite.tmp1")
        recomputed_pitch = U.call(cmd)

        pitch_error = 0
        vuv_error = 0
        for p, q in zip(pitch, recomputed_pitch):
            if p != 0 and q != 0:
                pitch_error += 16000 * np.abs(1 / p - 1 / q)
            elif not (p == 0 and q == 0):
                vuv_error += 1
        return pitch_error, vuv_error

    pitch_error_cc, vuv_error_cc = compute_error("excite.tmp2")
    pitch_error_py, vuv_error_py = compute_error("excite.tmp3")

    tol = 0
    assert pitch_error_py <= pitch_error_cc + tol
    tol = 5
    assert vuv_error_py <= vuv_error_cc + tol

    U.call("rm -f excite.tmp?", get=False)


@pytest.mark.parametrize("voiced_region", ["pulse", "sinusoidal", "sawtooth"])
def test_waveform(voiced_region, P=80, verbose=False):
    excite = diffsptk.ExcitationGeneration(
        P, voiced_region=voiced_region, unvoiced_region="zeros"
    )
    pitch = torch.from_numpy(
        U.call("x2x +sd tools/SPTK/asset/data.short | " f"pitch -s 16 -p {P} -o 0 -a 2")
    )
    e = excite(pitch)
    if verbose:
        sf.write(f"excite_{voiced_region}.wav", e, 16000)
