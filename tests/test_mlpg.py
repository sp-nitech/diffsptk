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
import torch

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "delta", [([-0.5, 0, 0.5], [1, -2, 1]), ([3, 0, 1, 2, 0], [-1])]
)
def test_compatibility(device, delta, T=100, D=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    tmp1 = "mlpg.tmp1"
    tmp2 = "mlpg.tmp2"
    tmp3 = "mlpg.tmp3"
    H = len(delta) + 1
    U.call(f"nrand -s 1 -l {T*H*D} > {tmp1}", get=False)
    U.call(f"step -l {T*H*D} > {tmp2}", get=False)
    U.call(f"merge -l {H*D} -L {H*D} {tmp1} {tmp2} > {tmp3}", get=False)

    mlpg = diffsptk.MaximumLikelihoodParameterGeneration(T, seed=delta).to(device)
    mean = torch.from_numpy(U.call(f"cat {tmp1}").reshape(1, T, H * D)).to(device)
    d = " ".join(["-d " + " ".join([str(x) for x in c]) for c in delta])
    y = U.call(f"mlpg -l {D} {d} -R 1 {tmp3}").reshape(-1, D)
    U.call(f"rm {tmp1} {tmp2} {tmp3}", get=False)
    U.check_compatibility(y, mlpg, mean)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, B=2, T=20, D=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    delta = diffsptk.Delta().to(device)
    mlpg = diffsptk.MLPG(T).to(device)
    x = torch.randn(B, T, D, requires_grad=True, device=device)
    U.check_differentiable(U.compose(mlpg, delta), x)
