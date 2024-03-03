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

import pytest

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("o", [0, 1, 2, 3])
def test_compatibility(
    device, o, M=4, C=10, L=32, sr=8000, lifter=20, f_min=300, floor=1, B=2
):
    spec = diffsptk.Spectrum(L, eps=0)
    mfcc = diffsptk.MFCC(
        M, C, L, sr, lifter=lifter, f_min=f_min, floor=floor, out_format=o
    )

    s = sr // 1000
    U.check_compatibility(
        device,
        [mfcc, spec],
        [],
        f"nrand -l {B*L}",
        f"mfcc -m {M} -n {C} -l {L} -s {s} -c {lifter} -L {f_min} -e {floor} -o {o}",
        [],
        dx=L,
        dy=M + (o if o <= 1 else o - 1),
    )

    U.check_differentiability(device, [mfcc, spec], [B, L])
