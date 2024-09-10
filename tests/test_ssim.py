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
from skimage.metrics import structural_similarity
import torch

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
def test_compatibility(device, module, B=1, T=100, D=100):
    if device == "cuda" and not torch.cuda.is_available():
        return

    ssim = U.choice(
        module,
        diffsptk.SSIM,
        diffsptk.functional.ssim,
        {},
        {"reduction": "mean", "dynamic_range": 1, "padding": "valid"},
        n_input=2,
    )
    if hasattr(ssim, "to"):
        ssim = ssim.to(device)

    x = torch.rand(B, T, D, device=device)
    y = torch.rand(B, T, D, device=device)

    s1 = structural_similarity(
        x.cpu().numpy(),
        y.cpu().numpy(),
        channel_axis=0,
        data_range=1,
        gaussian_weights=True,
        use_sample_covariance=False,
    )

    s2 = ssim(x, y).cpu().item()
    assert U.allclose(s1, s2)

    s3 = ssim(x, x).cpu().item()
    assert U.allclose(1, s3)

    U.check_differentiability(device, ssim, [(B, T, D), (B, T, D)])


def test_special_case(B=2, T=30, D=30):
    x = torch.rand(B, T, D)
    y = torch.rand(B, T, D)
    s = diffsptk.SSIM(reduction="none", padding="same", dynamic_range=1)(x, y)
    assert s.shape == x.shape

    x = torch.randn(B, T, D)
    y = torch.randn(B, T, D)
    s1 = diffsptk.SSIM(reduction="sum", padding="same", dynamic_range=None)(x, y)
    s2 = diffsptk.SSIM(reduction="mean", padding="same", dynamic_range=None)(x, y)
    assert U.allclose(s1, s2 * x.numel())
