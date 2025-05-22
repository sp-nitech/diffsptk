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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("filter_order", [99, 100])
def test_analysis_synthesis(device, filter_order, verbose=False):
    if device == "cuda" and not torch.cuda.is_available():
        return

    x, sr = diffsptk.read("assets/data.wav", device=device)

    oband = diffsptk.FractionalOctaveBandAnalysis(sr, filter_order=filter_order).to(
        device
    )
    y = oband(x)
    x_hat = y.sum(dim=1).squeeze(0)
    assert (x - x_hat).abs().sum() < 120

    if verbose:
        diffsptk.write("reconst.wav", x_hat, sr)
