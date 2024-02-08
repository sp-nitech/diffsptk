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
@pytest.mark.parametrize("quantizer", [0, 1])
def test_compatibility(device, module, quantizer, v=10, n_bit=3, L=10):
    dequantize = U.choice(
        module,
        diffsptk.InverseUniformQuantization,
        diffsptk.functional.dequantize,
        {},
        {"abs_max": v, "n_bit": n_bit, "quantizer": quantizer},
    )

    U.check_compatibility(
        device,
        dequantize,
        [],
        [f"ramp -s -1 -l {L}"],
        f"x2x +di | dequantize -v {v} -b {n_bit} -t {quantizer}",
        [],
    )

    U.check_differentiability(device, dequantize, [L])
