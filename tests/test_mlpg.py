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
@pytest.mark.parametrize(
    "seed",
    [
        [[-0.5, 0, 0.5], [1, -2, 1]],
        [[3, 0, 1, 2, 0], [-1]],
    ],
)
def test_compatibility(device, seed, T=100, D=2):
    mlpg = diffsptk.MaximumLikelihoodParameterGeneration(T, seed=seed)

    opt = " ".join(["-d " + " ".join([str(x) for x in c]) for c in seed])
    H = len(seed) + 1
    tmp1 = "mlpg.tmp1"
    tmp2 = "mlpg.tmp2"
    tmp3 = "mlpg.tmp3"
    U.check_compatibility(
        device,
        mlpg,
        [
            f"nrand -s 1 -l {T*D*H} > {tmp1}",  # mean
            f"step -l {T*D*H} > {tmp2}",  # unit variance
            f"merge -l {D*H} -L {D*H} {tmp1} {tmp2} > {tmp3}",
        ],
        f"cat {tmp1}",
        f"mlpg -l {D} {opt} -R 1 {tmp3}",
        [f"rm {tmp1} {tmp2} {tmp3}"],
        dx=D * H,
        dy=D,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, B=2, T=20, D=2):
    delta = diffsptk.Delta()
    mlpg = diffsptk.MLPG(T)
    U.check_differentiable(device, [mlpg, delta], [B, T, D])
