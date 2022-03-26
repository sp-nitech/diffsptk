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
@pytest.mark.parametrize("quantizer", [0, 1])
def test_compatibility(device, quantizer, v=3, n_bit=8, L=20):
    if device == "cuda" and not torch.cuda.is_available():
        return

    quantize = diffsptk.UniformQuantization(v, n_bit, quantizer).to(device)
    dequantize = diffsptk.InverseUniformQuantization(v, n_bit, quantizer).to(device)
    x = quantize(torch.from_numpy(U.call(f"nrand -l {L}")).to(device))
    cmd = (
        f"nrand -l {L} | "
        f"quantize -v {v} -b {n_bit} -t {quantizer} | "
        f"dequantize -v {v} -b {n_bit} -t {quantizer}"
    )
    y = U.call(cmd)
    U.check_compatibility(y, dequantize, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, L=20):
    if device == "cuda" and not torch.cuda.is_available():
        return

    dequantize = diffsptk.InverseUniformQuantization().to(device)
    x = torch.randn(L, requires_grad=True, device=device)
    U.check_differentiable(dequantize, x)
