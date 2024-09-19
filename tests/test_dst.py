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
from scipy.fft import dst as scipy_dst

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
def test_compatibility(device, module, dst_type, L=8, B=2):
    dst = U.choice(
        module,
        diffsptk.DST,
        diffsptk.functional.dst,
        {"dst_length": L},
        {"dst_type": dst_type},
    )

    def func(x):
        return scipy_dst(x, type=dst_type, norm="ortho")

    U.check_confidence(
        device,
        dst,
        func,
        [B, L],
    )

    U.check_differentiability(device, dst, [B, L])
