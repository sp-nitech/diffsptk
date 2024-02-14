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
@pytest.mark.parametrize("reduction", ["none", "mean"])
def test_compatibility(device, module, reduction, B=2, L=10):
    rmse = U.choice(
        module,
        diffsptk.RMSE,
        diffsptk.functional.rmse,
        {},
        {"reduction": reduction},
        n_input=2,
    )

    opt = "-f" if reduction == "none" else ""
    tmp1 = "rmse.tmp1"
    tmp2 = "rmse.tmp2"
    U.check_compatibility(
        device,
        rmse,
        [f"nrand -s 1 -l {B*L} > {tmp1}", f"nrand -s 2 -l {B*L} > {tmp2}"],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"rmse -l {L} {opt} {tmp1} {tmp2}",
        [f"rm {tmp1} {tmp2}"],
        dx=L,
    )

    U.check_differentiability(device, rmse, [(B, L), (B, L)])
