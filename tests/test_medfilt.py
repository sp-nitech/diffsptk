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
@pytest.mark.parametrize("K", [1, 2, 3])
@pytest.mark.parametrize("L", [1, 2])
@pytest.mark.parametrize("across_features", [False, True])
@pytest.mark.parametrize("magic_number", [None, 0])
def test_compatibility(
    device, dtype, module, K, L, across_features, magic_number, N=10
):
    medfilt = U.choice(
        module,
        diffsptk.MedianFilter,
        diffsptk.functional.medfilt,
        {
            "filter_length": K,
            "across_features": across_features,
            "magic_number": magic_number,
        },
    )

    w = 1 if across_features else 0
    opt = "" if magic_number is None else f"--magic {magic_number}"

    U.check_compatibility(
        device,
        dtype,
        medfilt,
        [],
        f"nrand -l {N * L} | sopr -ROUND",
        f"medfilt -l {L} -K {K} -w {w} {opt}",
        [],
        dx=L,
        dy=None if across_features else L,
    )

    U.check_differentiability(device, dtype, medfilt, [N])


def test_various_shape(K=3, N=10):
    medfilt = diffsptk.MedianFilter(K)
    U.check_various_shape(medfilt, [(N,), (N, 1), (1, N, 1)])
