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
@pytest.mark.parametrize("io_format", [0, 1])
def test_compatibility(device, dtype, module, io_format, M=12, L=32, B=2):
    root_pol = diffsptk.PolynomialToRoots(
        M, out_format=io_format, device=device, dtype=dtype
    )
    pol_root = U.choice(
        module,
        diffsptk.RootsToPolynomial,
        diffsptk.functional.pol_root,
        {"order": M, "in_format": io_format, "dtype": dtype},
    )

    U.check_compatibility(
        device,
        dtype,
        [pol_root, root_pol],
        [],
        f"nrand -l {B * L} | acorr -l {L} -m {M} -o 1",
        "cat",
        [],
        dx=M + 1,
        dy=M + 1,
    )

    U.check_differentiability(
        device, dtype, [torch.abs, pol_root, root_pol], [B, M + 1]
    )
