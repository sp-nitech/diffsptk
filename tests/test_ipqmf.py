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
@pytest.mark.parametrize("a", [10, 50, 100])
def test_compatibility(device, a, M, tau=0.01, eps=0.01, K=4, T=20):
    ipqmf = diffsptk.IPQMF(K, M, alpha=a, step_size=tau, eps=eps)

    U.check_compatibility(
        device,
        ipqmf,
        [],
        f"nrand -l {K*T}",
        f"transpose -r {K} -c {T} | ipqmf -k {K} -m {M} -a {a} -s {tau} -d {eps}",
        [],
        dx=T,
    )

    U.check_differentiability(device, ipqmf, [K, T], opt={"keepdim": False})


def test_learnable(K=4, M=10, T=20):
    ipqmf = diffsptk.IPQMF(K, M, learnable=True)
    U.check_learnable(ipqmf, (K, T))
