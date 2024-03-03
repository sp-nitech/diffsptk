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
def test_compatibility(device, module, m=4, M=5, L=16, B=2):
    c2acr = U.choice(
        module,
        diffsptk.CepstrumToAutocorrelation,
        diffsptk.functional.c2acr,
        {"cep_order": m},
        {"acr_order": M, "n_fft": L},
    )

    U.check_compatibility(
        device,
        c2acr,
        [],
        f"nrand -l {B*(m+1)}",
        f"c2acr -m {m} -M {M} -l {L}",
        [],
        dx=m + 1,
        dy=M + 1,
    )

    U.check_differentiability(device, c2acr, [B, m + 1])
