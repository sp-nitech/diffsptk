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


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("out_complex", [False, True])
def test_compatibility(
    device, dtype, module, out_complex, T=100, P=10, L1=12, L2=16, n=1, w=1, eps=1e-6
):
    stft = U.choice(
        module,
        diffsptk.STFT,
        diffsptk.functional.stft,
        {
            "frame_length": L1,
            "frame_period": P,
            "fft_length": L2,
            "window": w,
            "norm": n,
            "eps": eps,
            "out_format": "complex" if out_complex else "power",
            "device": device,
            "dtype": dtype,
        },
    )

    cmd = f"frame -l {L1} -p {P} | window -l {L1} -L {L2} -n {n} -w {w} | "
    if out_complex:
        cmd += f"fftr -l {L2} -H -o 3"
    else:
        cmd += f"spec -l {L2} -e {eps} -o 3"
    U.check_compatibility(
        device,
        dtype,
        [torch.abs, stft] if out_complex else stft,
        [],
        f"nrand -l {T}",
        cmd,
        [],
        dy=L2 // 2 + 1,
    )

    U.check_differentiability(device, dtype, [torch.abs, stft], [T])


@pytest.mark.parametrize("learnable", [True, ("basis",), ("window",)])
def test_learnable(learnable, P=10, L1=12, L2=16, T=80):
    stft = diffsptk.STFT(L1, P, L2, learnable=learnable)
    U.check_learnable(stft, (T,))
