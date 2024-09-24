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
from scipy.signal import hilbert2 as scipy_hilbert2

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("L", [(16, 8), None])
def test_compatibility(device, module, L):
    if module and L is None:
        return

    hilbert2 = U.choice(
        module,
        diffsptk.TwoDimensionalHilbertTransform,
        diffsptk.functional.hilbert2,
        {"fft_length": L},
    )

    def func(x):
        return scipy_hilbert2(x, N=L)

    if L is None:
        L = (16, 8)

    U.check_confidence(
        device,
        hilbert2,
        func,
        L,
    )

    U.check_differentiability(device, [lambda x: x.real, hilbert2], L)
