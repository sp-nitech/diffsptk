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
@pytest.mark.parametrize("in_norm", [False, True])
@pytest.mark.parametrize("out_norm", [False, True])
@pytest.mark.parametrize("in_mul", [False, True])
@pytest.mark.parametrize("out_mul", [False, True])
@pytest.mark.parametrize(
    "M, A, G", [[4, 0, 0.1], [4, 0, 0.2], [2, 0.1, 0.1], [6, 0.1, 0.2]]
)
def test_compatibility(
    device, in_norm, out_norm, in_mul, out_mul, M, A, G, m=4, a=0, g=0.1, B=2
):
    if device == "cuda" and not torch.cuda.is_available():
        return

    mgc2mgc = diffsptk.MelGeneralizedCepstrumToMelGeneralizedCepstrum(
        in_order=m,
        out_order=M,
        in_alpha=a,
        out_alpha=A,
        in_gamma=g,
        out_gamma=G,
        in_norm=in_norm,
        out_norm=out_norm,
        in_mul=in_mul,
        out_mul=out_mul,
    ).to(device)
    x = torch.from_numpy(
        U.call(f"nrand -l {B*(m+1)} | sopr -ABS").reshape(-1, m + 1)
    ).to(device)
    opt = f"-m {m} -M {M} -a {a} -A {A} -g {g} -G {G} "
    if in_norm:
        opt += "-n "
    if out_norm:
        opt += "-N "
    if in_mul:
        opt += "-u "
    if out_mul:
        opt += "-U "
    y = U.call(f"nrand -l {B*(m+1)} | sopr -ABS | mgc2mgc {opt}").reshape(-1, M + 1)
    U.check_compatibility(y, mgc2mgc, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("in_norm", [False, True])
@pytest.mark.parametrize("out_norm", [False, True])
@pytest.mark.parametrize("in_mul", [False, True])
@pytest.mark.parametrize("out_mul", [False, True])
def test_differentiable(
    device, in_norm, out_norm, in_mul, out_mul, m=9, M=9, a=0, A=0.3, g=0.2, G=0.2, B=2
):
    if device == "cuda" and not torch.cuda.is_available():
        return

    mgc2mgc = diffsptk.MelGeneralizedCepstrumToMelGeneralizedCepstrum(
        in_order=m,
        out_order=M,
        in_alpha=a,
        out_alpha=A,
        in_gamma=g,
        out_gamma=G,
        in_norm=in_norm,
        out_norm=out_norm,
        in_mul=in_mul,
        out_mul=out_mul,
    ).to(device)
    x = torch.randn(B, m + 1, requires_grad=True, device=device)
    U.check_differentiable(U.compose(mgc2mgc, torch.abs), x)
