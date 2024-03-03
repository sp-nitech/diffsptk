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
@pytest.mark.parametrize("out_format", [0, 1, 2])
def test_compatibility(device, module, out_format, L=5, B=2):
    entropy = U.choice(
        module,
        diffsptk.Entropy,
        diffsptk.functional.entropy,
        {},
        {"out_format": out_format},
    )

    U.check_compatibility(
        device,
        entropy,
        [],
        f"nrand -l {B*L} -d 0.5 | sopr -ABS",
        f"entropy -l {L} -o {out_format} -f",
        [],
        dx=L,
    )

    U.check_differentiability(device, [entropy, torch.abs], [B, L])
