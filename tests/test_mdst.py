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
@pytest.mark.parametrize("window", ["sine", "vorbis", "kbd", "rectangular"])
def test_compatibility(device, dtype, module, window, L=512):
    mdst_params = {
        "frame_length": L,
        "window": window,
        "device": device,
        "dtype": dtype,
    }
    mdst = U.choice(
        module,
        diffsptk.MDST,
        diffsptk.functional.mdst,
        mdst_params,
    )
    imdst = diffsptk.IMDST(**mdst_params)

    U.check_compatibility(
        device,
        dtype,
        [imdst, mdst],
        [],
        f"nrand -l {L}",
        "sopr",
        [],
    )

    U.check_differentiability(device, dtype, mdst, [L])
