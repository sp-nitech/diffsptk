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
def test_compatibility(device, module, L=8, B=2):
    dst = diffsptk.DST(L)
    idst = U.choice(
        module,
        diffsptk.IDST,
        diffsptk.functional.idst,
        {"dst_length": L},
    )

    U.check_compatibility(
        device,
        [idst, dst],
        [],
        f"nrand -l {B*L}",
        "sopr",
        [],
        dx=L,
        dy=L,
    )

    U.check_differentiability(device, idst, [B, L])
