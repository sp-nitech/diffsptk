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
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("gamma", [-1, 0, 1])
@pytest.mark.parametrize("use_power", [False, True])
def test_compatibility(
    device, module, gamma, use_power, C=12, L=32, sr=8000, f_min=30, f_max=4000, B=2
):
    fbank_params = {
        "n_channel": C,
        "fft_length": L,
        "sample_rate": sr,
        "f_min": f_min,
        "f_max": f_max,
        "gamma": gamma,
        "use_power": use_power,
    }
    fbank = diffsptk.FBANK(**fbank_params, out_format="y")
    ifbank = U.choice(
        module,
        diffsptk.IFBANK,
        diffsptk.functional.ifbank,
        fbank_params,
    )

    U.check_compatibility(
        device,
        [ifbank, fbank],
        [],
        f"nrand -l {B * L} | spec -l {L} -o 3",
        "sopr",
        [],
        dx=L // 2 + 1,
        dy=L // 2 + 1,
        eq=lambda a, b: U.allclose(a[..., 1:5], b[..., 1:5]),
    )

    U.check_differentiability(device, [ifbank, fbank, torch.abs], [B, L // 2 + 1])


def test_learnable(C=10, L=32, sr=8000):
    ifbank = diffsptk.InverseMelFilterBankAnalysis(
        n_channel=C,
        fft_length=L,
        sample_rate=sr,
        learnable=True,
    )
    U.check_learnable(ifbank, (C,))
