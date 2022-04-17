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
@pytest.mark.parametrize("M", [10, 11])
def test_compatibility(device, M, K=4, T=20):
    ipqmf = diffsptk.IPQMF(K, M)

    U.check_compatibility(
        device,
        ipqmf,
        [],
        f"nrand -l {K*T}",
        f"transpose -r {K} -c {T} | ipqmf -k {K} -m {M}",
        [],
        dx=T,
    )

    U.check_differentiable(device, ipqmf, [K, T])
