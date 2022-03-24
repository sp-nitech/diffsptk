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
def test_compatibility(device, C=10, L=32, sr=8000, f_min=300, f_max=3400, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    spec = diffsptk.Spectrum(L, eps=0).to(device)
    fbank = diffsptk.MelFilterBankAnalysis(
        C, L, sr, f_min=f_min, f_max=f_max, out_format="yE"
    ).to(device)
    x = spec(torch.from_numpy(U.call(f"nrand -l {B*L}").reshape(-1, L)).to(device))
    cmd = (
        f"nrand -l {B*L} | "
        f"fbank -n {C} -l {L} -s {sr//1000} -L {f_min} -H {f_max} -o 1"
    )
    y = U.call(cmd).reshape(-1, C + 1)
    U.check_compatibility(y, fbank, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, C=10, L=32, sr=8000, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    spec = diffsptk.Spectrum(L).to(device)
    fbank = diffsptk.MelFilterBankAnalysis(C, L, sr, out_format="yE").to(device)
    x = torch.randn(B, L, requires_grad=True, device=device)
    U.check_differentiable(U.compose(fbank, spec), x)