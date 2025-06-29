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
@pytest.mark.parametrize("reduction", ["none", "batchmean", "mean", "sum"])
@pytest.mark.parametrize("lag", [1, 0, -1])
def test_compatibility(device, dtype, module, reduction, lag, norm=2, T=20, L=5):
    flux = U.choice(
        module,
        diffsptk.Flux,
        diffsptk.functional.flux,
        {"lag": lag, "norm": norm, "reduction": reduction},
    )

    if reduction == "none":
        reduce_cmd = "cat"
    elif reduction == "batchmean":
        reduce_cmd = "average"
    elif reduction == "mean":
        reduce_cmd = f"average | sopr -d {L ** (1 / norm)}"
    elif reduction == "sum":
        reduce_cmd = "vsum"
    else:
        raise ValueError

    tmp1 = "flux.tmp1"
    tmp2 = "flux.tmp2"
    U.check_compatibility(
        device,
        dtype,
        flux,
        [f"nrand -s 1 -l {T * L} > {tmp1}", f"nrand -s 2 -l {T * L} > {tmp2}"],
        [f"cat {tmp1}", f"cat {tmp2}"],
        (
            f"delay -l {L} -s -{abs(lag)} {tmp1 if 0 < lag else tmp2} | "
            f"vopr -l {L} -s {tmp2 if 0 < lag else tmp1} | "
            f"sopr -p {norm} | vsum -l 1 -t {L} | sopr -p {1 / norm} | "
            f"{reduce_cmd}"
        ),
        [f"rm {tmp1} {tmp2}"],
        dx=L,
    )

    U.check_differentiability(device, dtype, flux, [(T, L), (T, L)])


@pytest.mark.parametrize("module", [False, True])
def test_special_case(module, T=20):
    flux = U.choice(
        module,
        diffsptk.Flux,
        diffsptk.functional.flux,
    )
    x = diffsptk.nrand(T)
    assert U.allclose(flux(x), flux(x, x))
