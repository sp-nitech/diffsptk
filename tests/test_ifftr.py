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
def test_compatibility(device, module, L=16, B=2):
    fftr = diffsptk.RealValuedFastFourierTransform(L)
    ifftr = U.choice(
        module,
        diffsptk.RealValuedInverseFastFourierTransform,
        diffsptk.functional.ifftr,
        {"fft_length": L, "learnable": "debug"},
    )

    U.check_compatibility(
        device,
        [ifftr, fftr],
        [],
        f"nrand -l {B * L}",
        "sopr",
        [],
        dx=L,
        dy=L,
    )

    U.check_differentiability(device, [ifftr, fftr], [B, L])


def test_learnable(L=16):
    ifftr = diffsptk.RealValuedInverseFastFourierTransform(L, learnable=True)
    U.check_learnable(ifftr, (L // 2 + 1,), dtype=U.get_complex_dtype())
