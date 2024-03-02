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
@pytest.mark.parametrize("in_norm", [False, True])
@pytest.mark.parametrize("out_norm", [False, True])
@pytest.mark.parametrize("in_mul", [False, True])
@pytest.mark.parametrize("out_mul", [False, True])
@pytest.mark.parametrize(
    "M, A, G", [[4, 0, 0.1], [4, 0, 0.2], [2, 0.1, 0.1], [6, 0.1, 0.2]]
)
def test_compatibility(
    device,
    module,
    in_norm,
    out_norm,
    in_mul,
    out_mul,
    M,
    A,
    G,
    m=4,
    a=0,
    g=0.1,
    L=256,
    B=2,
):
    mgc2mgc = U.choice(
        module,
        diffsptk.MelGeneralizedCepstrumToMelGeneralizedCepstrum,
        diffsptk.functional.mgc2mgc,
        {"in_order": m},
        {
            "out_order": M,
            "in_alpha": a,
            "out_alpha": A,
            "in_gamma": g,
            "out_gamma": G,
            "in_norm": in_norm,
            "out_norm": out_norm,
            "in_mul": in_mul,
            "out_mul": out_mul,
            "n_fft": L,
        },
    )

    opt1 = f"-m {m} -M {m} -a 0 -A {a} -g {0} -G {g} "
    if in_norm:
        opt1 += "-N "
    if in_mul:
        opt1 += "-U "

    opt2 = f"-m {m} -M {M} -a {a} -A {A} -g {g} -G {G} "
    if in_norm:
        opt2 += "-n "
    if out_norm:
        opt2 += "-N "
    if in_mul:
        opt2 += "-u "
    if out_mul:
        opt2 += "-U "

    U.check_compatibility(
        device,
        mgc2mgc,
        [],
        f"nrand -l {B*L} | fftcep -l {L} -m {m} | mgc2mgc {opt1}",
        f"mgc2mgc {opt2}",
        [],
        dx=m + 1,
        dy=M + 1,
    )

    U.check_differentiability(device, [mgc2mgc, torch.abs], [B, m + 1])
