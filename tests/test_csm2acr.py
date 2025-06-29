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


@pytest.mark.skip_float_check
@pytest.mark.parametrize("module", [False, True])
def test_compatibility(device, dtype, module, M=25, L=100, B=2):
    csm2acr = U.choice(
        module,
        diffsptk.CompositeSinusoidalModelCoefficientsToAutocorrelation,
        diffsptk.functional.csm2acr,
        {"acr_order": M, "device": device, "dtype": dtype},
    )

    U.check_compatibility(
        device,
        dtype,
        csm2acr,
        [],
        f"nrand -l {B * L} | acorr -m {M} -l {L} | acr2csm -m {M}",
        f"csm2acr -m {M}",
        [],
        dx=M + 1,
        dy=M + 1,
    )

    acorr = diffsptk.Autocorrelation(L, M)
    acr2csm = diffsptk.AutocorrelationToCompositeSinusoidalModelCoefficients(
        M, device=device, dtype=dtype
    )
    U.check_differentiability(device, dtype, [csm2acr, acr2csm, acorr], [B, L])
