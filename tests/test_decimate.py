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
def test_compatibility(device, module, P=2, S=1, T=20, L=4):
    decimate = U.choice(
        module,
        diffsptk.Decimation,
        diffsptk.functional.decimate,
        {},
        {"period": P, "start": S, "dim": 0},
    )

    U.check_compatibility(
        device,
        decimate,
        [],
        f"ramp -l {T*L}",
        f"decimate -l {L} -p {P} -s {S}",
        [],
        dx=L,
        dy=L,
    )

    U.check_differentiability(device, decimate, [T, L])
