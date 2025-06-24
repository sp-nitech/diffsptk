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

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("module", [False, True])
def test_compatibility(
    device,
    dtype,
    module,
    threshold=-40,
    ratio=2,
    attack_time=50,
    release_time=20,
    sr=16000,
    gain=0,
    T=20,
):
    drc = U.choice(
        module,
        diffsptk.DRC,
        diffsptk.functional.drc,
        {
            "threshold": threshold,
            "ratio": ratio,
            "attack_time": attack_time,
            "release_time": release_time,
            "sample_rate": sr,
            "makeup_gain": gain,
            "device": device,
            "dtype": dtype,
        },
    )

    U.check_compatibility(
        device,
        dtype,
        drc,
        [],
        "x2x +sd tools/SPTK/asset/data.short | sopr -d 32768",
        (
            f"drc -v 1 -t {threshold} -r {ratio} -A {attack_time} -R {release_time} "
            f"-s {sr // 1000} -m {gain} -d 0"
        ),
        [],
        eq=lambda a, b: np.corrcoef(a, b)[0, 1] > 0.99,
    )

    U.check_differentiability(device, dtype, drc, [T])


def test_learnable(T=20):
    drc = diffsptk.DRC(-20, 2, 50, 50, 16000, learnable=True)
    U.check_learnable(drc, (T,))
