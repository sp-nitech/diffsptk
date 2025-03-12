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
from scipy.fft import fft as scipy_fft

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("dht_type", [1, 2, 3, 4])
def test_convolution(device, module, dht_type, L=8, B=2):
    dht = U.choice(
        module,
        diffsptk.DHT,
        diffsptk.functional.dht,
        {"dht_length": L, "dht_type": dht_type},
    )

    if dht_type == 1:

        def func(x):
            X = scipy_fft(x, norm="ortho")
            return X.real - X.imag

        U.check_confidence(
            device,
            dht,
            func,
            [B, L],
        )

    U.check_differentiability(device, dht, [B, L])
