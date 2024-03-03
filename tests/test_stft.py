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


@pytest.mark.parametrize("module", [False, True])
def test_compatibility(module, T=100, P=10, L1=12, L2=16, n=1, w=1, eps=1e-6):
    stft = U.choice(
        module,
        diffsptk.STFT,
        diffsptk.functional.stft,
        {},
        {
            "frame_length": L1,
            "frame_period": P,
            "fft_length": L2,
            "window": w,
            "norm": n,
            "eps": eps,
            "out_format": "power",
        },
    )

    cmd = (
        f"frame -l {L1} -p {P} | "
        f"window -l {L1} -L {L2} -n {n} -w {w} |"
        f"spec -l {L2} -e {eps} -o 3"
    )
    U.check_compatibility(
        "cpu",
        stft,
        [],
        f"nrand -l {T}",
        cmd,
        [],
        dy=L2 // 2 + 1,
    )
