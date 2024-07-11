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

from operator import itemgetter

import pytest

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
def test_compatibility(device, module, T=19200):
    stft_params = {
        "frame_length": 400,
        "frame_period": 80,
        "fft_length": 512,
        "window": "hamming",
        "norm": "power",
    }
    stft = diffsptk.STFT(**stft_params, out_format="complex")
    istft = U.choice(
        module,
        diffsptk.ISTFT,
        diffsptk.functional.istft,
        {},
        stft_params,
    )

    U.check_compatibility(
        device,
        [itemgetter(slice(0, T)), istft, stft],
        [],
        "x2x +sd tools/SPTK/asset/data.short",
        "sopr",
        [],
    )

    U.check_differentiability(device, [istft, stft], [T])
