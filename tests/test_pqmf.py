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
    pqmf = diffsptk.PQMF(K, M, alpha=a, step_size=tau, eps=eps)

    U.check_compatibility(
        device,
        [lambda x: x.transpose(1, 2), pqmf],
        [],
        f"nrand -l {T}",
        f"pqmf -k {K} -m {M} -a {a} -s {tau} -d {eps}",
        [],
        dx=T,
        dy=K,
    )

    U.check_differentiability(device, pqmf, [T])


def test_various_shape(K=4, M=10, T=20):
    pqmf = diffsptk.PQMF(K, M)
    U.check_various_shape(pqmf, [(T,), (1, T), (1, 1, T)])


def test_learnable(K=4, M=10, T=20):
    pqmf = diffsptk.PQMF(K, M, learnable=True)
    U.check_learnable(pqmf, (T,))
