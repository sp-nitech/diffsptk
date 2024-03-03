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
def test_compatibility(device, M=9, alpha=0.1, beta=0.2, onset=2, L=128, B=2):
    mcpf = diffsptk.MelCepstrumPostfiltering(M, alpha, beta, onset, L)

    U.check_compatibility(
        device,
        mcpf,
        [],
        f"nrand -l {B*(M+1)}",
        f"mcpf -m {M} -a {alpha} -b {beta} -s {onset} -l {L}",
        [],
        dx=M + 1,
        dy=M + 1,
    )

    U.check_differentiability(device, mcpf, [B, M + 1])
