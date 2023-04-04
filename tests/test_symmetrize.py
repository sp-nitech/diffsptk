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
@pytest.mark.parametrize("in_format", [0, 1, 2, 3])
@pytest.mark.parametrize("out_format", [0, 1, 2, 3])
def test_compatibility(device, in_format, out_format, B=2, L=8):
    symmetrize = diffsptk.Symmetrization(in_format, out_format)

    if in_format == 0:
        dx = L // 2 + 1
    elif in_format == 1:
        dx = L
    elif in_format == 2:
        dx = L + 1
    elif in_format == 3:
        dx = L + 1

    if out_format == 0:
        dy = L // 2 + 1
    elif out_format == 1:
        dy = L
    elif out_format == 2:
        dy = L + 1
    elif out_format == 3:
        dy = L + 1

    U.check_compatibility(
        device,
        symmetrize,
        [],
        f"ramp -l {B*(L//2+1)} | symmetrize -l {L} -q 0 -o {in_format}",
        f"symmetrize -l {L} -q {in_format} -o {out_format}",
        [],
        dx=dx,
        dy=dy,
    )

    U.check_differentiable(device, symmetrize, [B, dx])
