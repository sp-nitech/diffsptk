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
import pytest

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("out_format", [0, 1, 2, 3])
def test_compatibility(device, out_format, P=80, sr=16000, L=1024, B=2):
    if torch.get_default_dtype() == torch.float:
        pytest.skip("This test is only for torch.double")

    pitch_spec = diffsptk.CheapTrick(P, L, sr, out_format=out_format)

    ksr = sr // 1000
    tmp1 = "pitch_spec.tmp1"
    tmp2 = "pitch_spec.tmp2"
    U.check_compatibility(
        device,
        pitch_spec,
        [
            f"x2x +sd tools/SPTK/asset/data.short > {tmp1}",
            f"pitch -s {ksr} -p {P} -L 80 -H 180 -o 1 {tmp1} > {tmp2}",
        ],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"pitch_spec -x -s {ksr} -p {P} -l {L} -q 1 -o {out_format} {tmp2} < {tmp1}",
        [f"rm {tmp1} {tmp2}"],
        dy=L // 2 + 1,
        rtol=1e-5 if device == "cpu" else 1e-4,
    )

    U.check_differentiability(
        device, pitch_spec, [(B, sr), (B, sr // P)], checks=[True, False]
    )
