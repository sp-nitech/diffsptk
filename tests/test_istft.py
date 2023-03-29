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
def test_compatibility(device):
    if device == "cuda" and not torch.cuda.is_available():
        return

    stft_params = {
        "frame_length": 400,
        "frame_period": 80,
        "fft_length": 512,
        "norm": "power",
        "window": "hamming",
    }
    stft = diffsptk.STFT(**stft_params, out_format="complex").to(device)
    istft = diffsptk.ISTFT(**stft_params).to(device)

    x = torch.from_numpy(U.call("x2x +sd tools/SPTK/asset/data.short")).to(device)
    y = istft(stft(x), out_length=x.size(0))
    assert torch.allclose(x, y)
