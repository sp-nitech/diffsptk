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
from scipy.fft import dct as scipy_dct

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
def test_compatibility(device, dtype, module, dct_type, L=8, B=2):
    dct = U.choice(
        module,
        diffsptk.DCT,
        diffsptk.functional.dct,
        {"dct_length": L, "dct_type": dct_type, "device": device, "dtype": dtype},
    )

    if dct_type == 2:
        U.check_compatibility(
            device,
            dtype,
            dct,
            [],
            f"nrand -l {B * L}",
            f"dct -l {L}",
            [],
            dx=L,
            dy=L,
        )

    def func(x):
        return scipy_dct(x, type=dct_type, norm="ortho")

    U.check_confidence(
        device,
        dtype,
        dct,
        func,
        [B, L],
    )

    U.check_differentiability(device, dtype, dct, [B, L])
