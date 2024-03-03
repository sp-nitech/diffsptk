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
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("M", [0, 1, 7, 8])
@pytest.mark.parametrize("out_format", [0, 1, 2, 3])
def test_compatibility(device, module, M, out_format, L=32, B=2):
    lpc2lsp = U.choice(
        module,
        diffsptk.LinearPredictiveCoefficientsToLineSpectralPairs,
        diffsptk.functional.lpc2lsp,
        {"lpc_order": M},
        {"log_gain": True, "sample_rate": 8000, "out_format": out_format},
    )

    U.check_compatibility(
        device,
        lpc2lsp,
        [],
        f"nrand -l {B*L} | lpc -l {L} -m {M}",
        f"lpc2lsp -m {M} -o {out_format} -k 1 -s 8",
        [],
        dx=M + 1,
        dy=M + 1,
    )

    U.check_differentiability(device, lpc2lsp, [B, M + 1])
