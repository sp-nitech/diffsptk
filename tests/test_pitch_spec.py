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

import pylstraight
import pytest
import torch

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("out_format", [0, 1, 2, 3])
def test_compatibility_d4c(device, out_format, P=80, sr=16000, L=1024, B=2):
    if torch.get_default_dtype() == torch.float:
        pytest.skip("This test is only for torch.double.")

    pitch_spec = diffsptk.PitchAdaptiveSpectralAnalysis(
        P, sr, L, algorithm="cheap-trick", out_format=out_format
    )

    s = sr // 1000
    tmp1 = "pitch_spec.tmp1"
    tmp2 = "pitch_spec.tmp2"
    U.check_compatibility(
        device,
        pitch_spec,
        [
            f"x2x +sd tools/SPTK/asset/data.short > {tmp1}",
            f"pitch -s {s} -p {P} -L 80 -H 180 -o 1 {tmp1} > {tmp2}",
        ],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"pitch_spec -x -s {s} -p {P} -l {L} -q 1 -o {out_format} {tmp2} < {tmp1}",
        [f"rm {tmp1} {tmp2}"],
        dy=L // 2 + 1,
        rtol=1e-5 if device == "cpu" else 1e-4,
    )

    U.check_differentiability(
        device, pitch_spec, [(B, sr), (B, sr // P)], checks=[True, False]
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("out_format", [0, 1, 2, 3])
def test_compatibility_straight(device, out_format, P=80, sr=16000, L=2048, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    if torch.get_default_dtype() == torch.float:
        pytest.skip("This test is only for torch.double.")

    cmd = "x2x +sd tools/SPTK/asset/data.short"
    x = U.call(cmd)

    s = sr // 1000
    cmd = f"x2x +sd tools/SPTK/asset/data.short | pitch -s {s} -p {P} -L 80 -H 180 -o 1"
    f0 = U.call(cmd)

    pitch_spec = diffsptk.PitchAdaptiveSpectralAnalysis(
        P, sr, L, algorithm="straight", out_format=out_format
    ).to(device)

    sp_formats = {0: "db", 1: "log", 2: "linear", 3: "power"}
    sp = pylstraight.extract_sp(
        x, sr, f0, frame_shift=P / s, sp_format=sp_formats[out_format]
    )

    sp_hat = (
        pitch_spec(torch.from_numpy(x).to(device), torch.from_numpy(f0).to(device))
        .cpu()
        .numpy()
    )

    assert U.allclose(sp, sp_hat, rtol=1e-1)

    U.check_differentiability(
        device, pitch_spec, [(B, sr), (B, sr // P)], checks=[True, False]
    )
