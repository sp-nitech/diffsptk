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
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("gamma, c", [(0, None), (1, None), (0, 2)])
def test_compatibility(device, module, gamma, c, M=4, B=2):
    ignorm = U.choice(
        module,
        diffsptk.GeneralizedCepstrumInverseGainNormalization,
        diffsptk.functional.ignorm,
        {"cep_order": M},
        {"gamma": gamma, "c": c},
    )

    opt = f"-g {gamma}" if c is None else f"-c {c}"
    U.check_compatibility(
        device,
        ignorm,
        [],
        f"nrand -l {B*(M+1)} | sopr -ABS",
        f"ignorm -m {M} {opt}",
        [],
        dx=M + 1,
        dy=M + 1,
    )

    U.check_differentiability(device, [ignorm, torch.abs], [B, M + 1])
