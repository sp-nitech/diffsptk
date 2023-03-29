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
def test_compatibility(
    device, fl=2048, fp=80, sr=22050, lag_min=22, lag_max=2047, n_bin=20, B=2
):
    if device == "cuda" and not torch.cuda.is_available():
        return

    frame = diffsptk.Frame(fl, fp, center=False).to(device)
    yingram = diffsptk.Yingram(fl, sr, lag_min, lag_max, n_bin).to(device)

    url = "https://raw.githubusercontent.com/revsic/torch-nansy/main/nansy/yingram.py"
    U.call(f"curl -s {url} > tmp.py", get=False)
    from tmp import Yingram as Target

    target = Target(fp, fl, lag_min, lag_max, n_bin, sr).to(device)
    U.call("rm -f tmp.py", get=False)

    x = diffsptk.nrand(B, sr).to(device)
    y = target(x)
    y_hat = yingram(frame(x))
    assert torch.allclose(y, y_hat)

    U.check_differentiable(device, yingram, [B, fl])
