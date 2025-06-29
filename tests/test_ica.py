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

import os

import numpy as np
import pytest
import torch

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("func", ["logcosh", "gauss"])
@pytest.mark.parametrize("batch_size", [None, 100])
def test_convergence(device, dtype, func, batch_size, T=1000, verbose=False):
    s = torch.stack(
        [
            diffsptk.functional.excite(
                torch.tensor([200.0]),
                frame_period=T,
                voiced_region="sinusoidal",
                unvoiced_region="zeros",
            ),
            diffsptk.functional.excite(
                torch.tensor([140.0]),
                frame_period=T,
                voiced_region="triangle",
                unvoiced_region="zeros",
            ),
            diffsptk.functional.excite(
                torch.tensor([0.0]),
                frame_period=T,
                unvoiced_region="gauss",
            ),
        ],
        dim=1,
    ).to(device=device, dtype=dtype)
    K = s.shape[1]

    A = torch.tensor(
        [
            [0.8, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.3, 0.3, 0.5],
        ]
    ).to(device=device, dtype=dtype)
    x = torch.matmul(s, A.T)
    M = x.shape[1] - 1

    ica = diffsptk.ICA(
        M,
        K,
        func=func,
        batch_size=batch_size,
        verbose=verbose,
        device=device,
        dtype=dtype,
    )
    ica(x)
    p = ica.transform(x)

    r = np.corrcoef(s.T.cpu().numpy(), p.T.cpu().numpy())
    np.fill_diagonal(r, 0)
    assert np.all(np.max(np.abs(r), axis=1) > 0.98)

    if verbose:
        tmp = "source.dat"
        s.T.cpu().numpy().astype(np.float64).tofile(tmp)
        cmd = (
            f"./tools/SPTK/tools/venv/bin/python ./tools/SPTK/bin/gwave {tmp} "
            f"source.png -i {K} -n {T} -g -r"
        )
        U.call(cmd, get=False)
        os.remove(tmp)

        tmp = "observed.dat"
        x.T.cpu().numpy().astype(np.float64).tofile(tmp)
        cmd = (
            f"./tools/SPTK/tools/venv/bin/python ./tools/SPTK/bin/gwave {tmp} "
            f"observed.png -i {M + 1} -n {T} -g -r"
        )
        U.call(cmd, get=False)
        os.remove(tmp)

        tmp = "predicted.dat"
        p.T.cpu().numpy().astype(np.float64).tofile(tmp)
        cmd = (
            f"./tools/SPTK/tools/venv/bin/python ./tools/SPTK/bin/gwave {tmp} "
            f"predicted.png -i {K} -n {T} -g -r"
        )
        U.call(cmd, get=False)
        os.remove(tmp)
