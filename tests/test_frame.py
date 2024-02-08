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
@pytest.mark.parametrize("fl", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("fp", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("zmean", [False, True])
def test_compatibility(device, module, fl, fp, center, zmean, T=20):
    frame = U.choice(
        module,
        diffsptk.Frame,
        diffsptk.functional.frame,
        {},
        {"frame_length": fl, "frame_period": fp, "center": center, "zmean": zmean},
    )

    opt = "-z" if zmean else ""
    n = 0 if center else 1
    U.check_compatibility(
        device,
        frame,
        [],
        f"ramp -l {T}",
        f"frame -l {fl} -p {fp} -n {n} {opt}",
        [],
        dy=fl,
    )

    U.check_differentiability(device, frame, [T], check_zero_grad=not zmean)
