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


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("M", [0, 7, 8])
@pytest.mark.parametrize("n_iter", [0, 3])
def test_compatibility(device, dtype, module, M, n_iter, L=32, B=2, alpha=0.1):
    spec = diffsptk.Spectrum(L, eps=0)
    mcep = U.choice(
        module,
        diffsptk.MelCepstralAnalysis,
        diffsptk.functional.mcep,
        {
            "fft_length": L,
            "cep_order": M,
            "n_iter": n_iter,
            "alpha": alpha,
            "device": device,
            "dtype": dtype,
        },
    )

    U.check_compatibility(
        device,
        dtype,
        [mcep, spec],
        [],
        f"nrand -l {B * L} | sopr -SQR",
        f"mgcep -i {n_iter} -l {L} -m {M} -a {alpha}",
        [],
        dx=L,
        dy=M + 1,
    )

    U.check_differentiability(device, dtype, [mcep, spec], [B, L])
