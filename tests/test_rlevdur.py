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
def test_compatibility(device, module, M=30, L=52, B=2):
    rlevdur = U.choice(
        module,
        diffsptk.ReverseLevinsonDurbin,
        diffsptk.functional.rlevdur,
        {"lpc_order": M},
    )

    U.check_compatibility(
        device,
        rlevdur,
        [],
        f"nrand -l {B*L} | lpc -m {M} -l {L}",
        f"rlevdur -m {M}",
        [],
        dx=M + 1,
        dy=M + 1,
    )

    lpc = diffsptk.LPC(L, M)
    U.check_differentiability(device, [rlevdur, lpc], [B, L])
