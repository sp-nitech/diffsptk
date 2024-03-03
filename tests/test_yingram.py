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
def test_compatibility(
    device, module, fl=2048, fp=80, sr=22050, lag_min=22, n_bin=20, B=2
):
    if device == "cuda" and not torch.cuda.is_available():
        return
    if torch.get_default_dtype() != torch.double:  # pragma: no cover
        return  # This is due to the difference in the calculation of autocorrelation.

    frame = diffsptk.Frame(fl, fp, center=False).to(device)
    yingram = U.choice(
        module,
        diffsptk.Yingram,
        diffsptk.functional.yingram,
        {"frame_length": fl},
        {"sample_rate": sr, "lag_min": lag_min, "n_bin": n_bin},
    )
    if module:
        yingram = yingram.to(device)

    url = "https://raw.githubusercontent.com/revsic/torch-nansy/main/nansy/yingram.py"
    U.call(f"curl -s {url} > tmp.py", get=False)
    from tmp import Yingram as Target

    target = Target(fp, fl, lag_min, fl - 1, n_bin, sr).to(device)
    U.call("rm -f tmp.py", get=False)

    x = diffsptk.nrand(B, sr).to(device)
    y = target(x)
    y_hat = yingram(frame(x))
    assert U.allclose(y, y_hat)

    U.check_differentiability(device, yingram, [B, fl])
