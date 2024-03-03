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
def test_compatibility(device, module, L=32, M=9, rate=0.8, B=2):
    lspcheck = U.choice(
        module,
        diffsptk.LineSpectralPairsStabilityCheck,
        diffsptk.functional.lspcheck,
        {"lsp_order": M},
        {"rate": rate, "n_iter": 1, "warn_type": "ignore"},
    )

    U.check_compatibility(
        device,
        lspcheck,
        [],
        f"nrand -l {B*L} | lpc -l {L} -m {M} | lpc2lsp -m {M}",
        f"lspcheck -m {M} -r {rate} -e 0 -x",
        [],
        dx=M + 1,
        dy=M + 1,
    )

    U.check_differentiability(
        device, [lspcheck, lambda x: torch.sort(x)[0], torch.abs], [B, M + 1]
    )
