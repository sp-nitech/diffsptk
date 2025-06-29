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
@pytest.mark.parametrize("dht_type", [1, 2, 3, 4])
def test_compatibility(device, dtype, module, dht_type, L=8, B=2):
    dht_params = {
        "dht_length": L,
        "dht_type": dht_type,
        "device": device,
        "dtype": dtype,
    }
    dht = diffsptk.DHT(**dht_params)
    idht = U.choice(
        module,
        diffsptk.IDHT,
        diffsptk.functional.idht,
        dht_params,
    )

    U.check_compatibility(
        device,
        dtype,
        [idht, dht],
        [],
        f"nrand -l {B * L}",
        "sopr",
        [],
        dx=L,
        dy=L,
    )

    U.check_differentiability(device, dtype, idht, [B, L])
