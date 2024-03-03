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
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("seed", [[[-1, 1], [-1, 0]], [3, 2]])
def test_compatibility(device, module, seed, T=10, L=2):
    delta = U.choice(
        module,
        diffsptk.Delta,
        diffsptk.functional.delta,
        {},
        {"seed": seed},
    )

    if U.is_array(seed[0]):
        opt = " ".join(["-d " + " ".join([str(w) for w in window]) for window in seed])
    else:
        opt = "-r " + " ".join([str(width) for width in seed])

    H = len(seed) + 1
    U.check_compatibility(
        device,
        delta,
        [],
        f"nrand -l {T*L}",
        f"delta -l {L} {opt}",
        [],
        dx=L,
        dy=L * H,
    )

    U.check_differentiability(device, delta, [1, T, L])
