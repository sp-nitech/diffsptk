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
@pytest.mark.parametrize("M", [7, 8])
@pytest.mark.parametrize("out_format", [0, 1, 2, 3])
def test_compatibility(device, module, M, out_format, L=16, B=2):
    lsp2sp = U.choice(
        module,
        diffsptk.LineSpectralPairsToSpectrum,
        diffsptk.functional.lsp2sp,
        {"lsp_order": M},
        {"fft_length": L, "out_format": out_format},
    )

    U.check_compatibility(
        device,
        lsp2sp,
        [],
        f"nrand -l {B*L} | lpc -l {L} -m {M} | lpc2lsp -m {M}",
        f"mglsp2sp -m {M} -l {L} -o {out_format}",
        [],
        dx=M + 1,
        dy=L // 2 + 1,
    )

    U.check_differentiability(device, [lsp2sp, torch.abs], [B, M + 1])
