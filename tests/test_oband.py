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


@pytest.mark.parametrize("filter_order", [99, 100])
@pytest.mark.parametrize("n_fract", [1, 2, 3])
def test_analysis_synthesis(device, dtype, filter_order, n_fract, verbose=False):
    x, sr = diffsptk.read("assets/data.wav", device=device, dtype=dtype)

    oband = diffsptk.FractionalOctaveBandAnalysis(
        sr, filter_order=filter_order, n_fract=n_fract, device=device, dtype=dtype
    )
    y = oband(x)
    x_hat = y.sum(dim=1).squeeze(0)
    if verbose:
        diffsptk.write("reconst.wav", x_hat, sr)

    assert (x - x_hat).abs().sum() < 120


def test_various_shape(sr=16000, T=1000):
    oband = diffsptk.FractionalOctaveBandAnalysis(sr)
    U.check_various_shape(oband, [(T,), (1, T), (1, 1, T)])
