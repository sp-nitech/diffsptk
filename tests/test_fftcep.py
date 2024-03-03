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
@pytest.mark.parametrize("M", [7, 8])
@pytest.mark.parametrize("n_iter", [0, 3])
def test_compatibility(device, module, M, n_iter, L=16, B=2, accel=0.001):
    spec = diffsptk.Spectrum(L, eps=0)
    fftcep = U.choice(
        module,
        diffsptk.CepstralAnalysis,
        diffsptk.functional.fftcep,
        {"fft_length": L},
        {
            "cep_order": M,
            "n_iter": n_iter,
            "accel": accel,
        },
    )

    U.check_compatibility(
        device,
        [fftcep, spec],
        [],
        f"nrand -l {B*L}",
        f"fftcep -i {n_iter} -l {L} -m {M} -a {accel}",
        [],
        dx=L,
        dy=M + 1,
    )

    U.check_differentiability(device, [fftcep, spec], [B, L])
