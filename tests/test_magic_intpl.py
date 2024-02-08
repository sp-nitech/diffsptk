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
import torch.nn.functional as F

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
def test_compatibility(device, module, N=10, L=2, magic_number=0):
    magic_intpl = U.choice(
        module,
        diffsptk.MagicNumberInterpolation,
        diffsptk.functional.magic_intpl,
        {},
        {"magic_number": magic_number},
    )

    U.check_compatibility(
        device,
        magic_intpl,
        [],
        "echo 0 9 0 0 0 0 2 1 0 0 4 5 0 0 | x2x +ad",
        f"magic_intpl -l {L} -magic {magic_number}",
        [],
        dx=L,
        dy=L,
    )

    U.check_differentiability(device, magic_intpl, [N, L])
    U.check_differentiability(device, [magic_intpl, F.dropout], [N, L])


def test_various_shape(N=10):
    magic_intpl = diffsptk.MagicNumberInterpolation()
    U.check_various_shape(magic_intpl, [(N,), (N, 1), (1, N, 1)], preprocess=F.dropout)
