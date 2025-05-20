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
@pytest.mark.parametrize("out_format", [0, 1, 2, 3])
def test_compatibility(
    device,
    module,
    out_format,
    M=4,
    C=10,
    L=32,
    sr=8000,
    lifter=20,
    floor=1,
    factor=0.3,
    B=2,
):
    spec = diffsptk.Spectrum(L, eps=0)
    plp = U.choice(
        module,
        diffsptk.PerceptualLinearPredictiveCoefficientsAnalysis,
        diffsptk.functional.plp,
        {
            "fft_length": L,
            "plp_order": M,
            "n_channel": C,
            "sample_rate": sr,
            "compression_factor": factor,
            "lifter": lifter,
            "floor": floor,
            "out_format": out_format,
        },
    )

    o = out_format
    s = sr // 1000
    U.check_compatibility(
        device,
        [plp, spec],
        [],
        f"nrand -l {B * L}",
        f"plp -m {M} -n {C} -l {L} -s {s} -c {lifter} -e {floor} -f {factor} -o {o}",
        [],
        dx=L,
        dy=M + (o if o <= 1 else o - 1),
    )

    U.check_differentiability(device, [plp, spec], [B, L])


def test_learnable(M=4, C=10, L=32, sr=8000):
    plp = diffsptk.PerceptualLinearPredictiveCoefficientsAnalysis(
        fft_length=L,
        plp_order=M,
        n_channel=C,
        sample_rate=sr,
        learnable=True,
    )
    U.check_learnable(plp, (L // 2 + 1,))
