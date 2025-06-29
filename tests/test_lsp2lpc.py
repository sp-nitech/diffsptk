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


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("M", [0, 1, 7, 8])
@pytest.mark.parametrize("in_format", [0, 1, 2, 3])
def test_compatibility(device, dtype, module, M, in_format, sr=8000, L=32, B=2):
    lsp2lpc = U.choice(
        module,
        diffsptk.LineSpectralPairsToLinearPredictiveCoefficients,
        diffsptk.functional.lsp2lpc,
        {
            "lpc_order": M,
            "log_gain": True,
            "sample_rate": sr,
            "in_format": in_format,
            "device": device,
            "dtype": dtype,
        },
    )

    s = sr // 1000
    U.check_compatibility(
        device,
        dtype,
        lsp2lpc,
        [],
        (
            f"nrand -l {B * L} | lpc -l {L} -m {M} | "
            f"lpc2lsp -m {M} -k 1 -s {s} -o {in_format}"
        ),
        f"lsp2lpc -m {M} -k 1 -s {s} -q {in_format}",
        [],
        dx=M + 1,
        dy=M + 1,
    )

    U.check_differentiability(device, dtype, lsp2lpc, [B, M + 1])
