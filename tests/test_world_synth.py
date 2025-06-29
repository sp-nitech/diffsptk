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

import diffsptk
import tests.utils as U


def test_compatibility(device, dtype, P=80, sr=16000, L=1024, B=2):
    world_synth = diffsptk.WorldSynthesis(P, sr, L, device=device, dtype=dtype)

    s = sr // 1000
    n = L // 2 + 1
    tmp1 = "world_synth.tmp1"
    tmp2 = "world_synth.tmp2"
    tmp3 = "world_synth.tmp3"
    tmp4 = "world_synth.tmp4"
    U.check_compatibility(
        device,
        dtype,
        world_synth,
        [
            f"x2x +sd tools/SPTK/asset/data.short > {tmp1}",
            f"pitch -s {s} -p {P} -L 80 -H 180 -o 1 -a 3 {tmp1} > {tmp2}",
            f"ap -s {s} -p {P} -l {L} -q 1 -o 0 {tmp2} {tmp1} > {tmp3}",
            f"pitch_spec -s {s} -p {P} -l {L} -q 1 -o 3 {tmp2} {tmp1} > {tmp4}",
        ],
        [f"cat {tmp2}", f"cat {tmp3}", f"cat {tmp4}"],
        f"world_synth -s {s} -p {P} -l {L} -S 3 -A 0 -F 1 {tmp4} {tmp3} < {tmp2}",
        [f"rm {tmp1} {tmp2} {tmp3} {tmp4}"],
        dx=[None, n, n],
        eq=lambda a, b: np.corrcoef(a, b)[0, 1] > 0.95,
    )

    U.check_differentiability(
        device,
        dtype,
        world_synth,
        [(B, sr // P), (B, sr // P, n), (B, sr // P, n)],
        scales=[80, 1, 1],
        checks=[False, True, True],
    )
