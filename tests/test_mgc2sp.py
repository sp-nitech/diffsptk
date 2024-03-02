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
@pytest.mark.parametrize("out_format", [0, 1, 2, 3, 4, 5, 6])
def test_compatibility(device, module, out_format, M=7, L=16, B=2):
    mgc2sp = U.choice(
        module,
        diffsptk.MelGeneralizedCepstrumToSpectrum,
        diffsptk.functional.mgc2sp,
        {"cep_order": M},
        {"fft_length": L, "out_format": out_format},
    )

    U.check_compatibility(
        device,
        mgc2sp,
        [],
        f"nrand -l {B*L} | fftcep -l {L} -m {M}",
        f"mgc2sp -m {M} -l {L} -o {out_format}",
        [],
        dx=M + 1,
        dy=L // 2 + 1,
    )

    U.check_differentiability(device, mgc2sp, [B, M + 1])
