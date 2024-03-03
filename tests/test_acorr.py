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
@pytest.mark.parametrize("M", [12, 13])
@pytest.mark.parametrize("out_format", [0, 1, 2, 3])
def test_compatibility(device, module, M, out_format, L=14, B=2):
    acorr = U.choice(
        module,
        diffsptk.Autocorrelation,
        diffsptk.functional.acorr,
        {"frame_length": L},
        {"acr_order": M, "norm": out_format == 1, "estimator": out_format},
    )

    U.check_compatibility(
        device,
        acorr,
        [],
        f"nrand -l {B*L}",
        f"acorr -l {L} -m {M} -o {out_format}",
        [],
        dx=L,
        dy=M + 1,
    )

    U.check_differentiability(device, acorr, [B, L])
