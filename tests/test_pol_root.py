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
def test_compatibility(device, M=12, L=32, B=2):
    root_pol = diffsptk.PolynomialToRoots(M)
    pol_root = diffsptk.RootsToPolynomial(M)

    U.check_compatibility(
        device,
        [pol_root, root_pol],
        [],
        f"nrand -l {B*L} | acorr -l {L} -m {M} -o 1",
        "cat",
        [],
        dx=M + 1,
        dy=M + 1,
    )

    U.check_differentiable(device, [pol_root, root_pol], [B, M + 1])
