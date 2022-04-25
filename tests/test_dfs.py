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


@pytest.mark.parametrize("device", ["cpu"])
def test_compatibility(device, b=[-0.42, 1], a=[1, -0.42], T=100):
    dfs = diffsptk.IIR(b, a, 30)

    bb = " ".join([str(x) for x in b])
    aa = " ".join([str(x) for x in a])

    U.check_compatibility(
        device,
        dfs,
        [],
        f"nrand -l {T}",
        f"dfs -b {bb} -a {aa}",
        [],
    )

    U.check_differentiable(device, dfs, [T])


def test_various_shape(T=10):
    pqmf = diffsptk.IIR()
    U.check_various_shape(pqmf, [(T,), (1, T), (1, 1, T)])
