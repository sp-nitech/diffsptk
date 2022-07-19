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
@pytest.mark.parametrize("fl", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("fp", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("center", [True, False])
def test_compatibility(device, fl, fp, center, T=20):
    frame = diffsptk.Frame(fl, fp, center=center, zmean=True)

    n = 0 if center else 1
    U.check_compatibility(
        device,
        frame,
        [],
        f"ramp -l {T}",
        f"frame -l {fl} -p {fp} -n {n} -z",
        [],
        dy=fl,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, fl=5, fp=3, B=4, T=20):
    frame = diffsptk.Frame(fl, fp, zmean=False)
    U.check_differentiable(device, frame, [B, T])
