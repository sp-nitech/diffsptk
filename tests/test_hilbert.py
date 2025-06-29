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
from scipy.signal import hilbert as scipy_hilbert

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("L", [7, 8, None])
def test_compatibility(device, dtype, module, L, B=2):
    if module and L is None:
        return

    hilbert = U.choice(
        module,
        diffsptk.HilbertTransform,
        diffsptk.functional.hilbert,
        {"fft_length": L, "device": device, "dtype": dtype},
    )

    def func(x):
        return scipy_hilbert(x, N=L)

    if L is None:
        L = 8

    U.check_confidence(
        device,
        dtype,
        hilbert,
        func,
        [B, L],
    )

    U.check_differentiability(device, dtype, [lambda x: x.real, hilbert], [B, L])
