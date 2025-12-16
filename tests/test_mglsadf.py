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


@pytest.mark.parametrize("ignore_gain", [False, True])
@pytest.mark.parametrize(
    "mode", ["multi-stage", "single-stage", "freq-domain", "pade-approx"]
)
@pytest.mark.parametrize("c", [0, 2])
def test_compatibility(
    device,
    dtype,
    ignore_gain,
    mode,
    c,
    alpha=0.42,
    M=24,
    P=80,
    L=400,
    fft_length=512,
    B=2,
):
    if mode == "multi-stage":
        params = {"taylor_order": 7, "cep_order": 100}
    elif mode == "single-stage":
        params = {"ir_length": 200, "n_fft": 512}
    elif mode == "freq-domain":
        params = {
            "frame_length": fft_length,
            "fft_length": fft_length,
            "window": "hamming",
        }
    elif mode == "pade-approx":
        params = {"pade_order": 5, "cep_order": 100}

    mglsadf = diffsptk.MLSA(
        M,
        P,
        alpha=alpha,
        c=c,
        ignore_gain=ignore_gain,
        phase="minimum",
        mode=mode,
        device=device,
        dtype=dtype,
        **params,
    )

    tmp1 = "mglsadf.tmp1"
    tmp2 = "mglsadf.tmp2"
    T = os.path.getsize("tools/SPTK/asset/data.short") // 2
    cmd1 = f"nrand -l {T} > {tmp1}"
    cmd2 = (
        f"x2x +sd tools/SPTK/asset/data.short | "
        f"frame -p {P} -l {L} | "
        f"window -w 1 -n 1 -l {L} -L {fft_length} | "
        f"mgcep -c {c} -a {alpha} -m {M} -l {fft_length} -E -60 > {tmp2}"
    )
    opt = "-k" if ignore_gain else ""
    U.check_compatibility(
        device,
        dtype,
        mglsadf,
        [cmd1, cmd2],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"mglsadf {tmp2} < {tmp1} -m {M} -p {P} -c {c} -a {alpha} {opt}",
        [f"rm {tmp1} {tmp2}"],
        dx=[None, M + 1],
        eq=lambda a, b: np.corrcoef(a, b)[0, 1] > 0.98,
    )

    S = T // 10
    U.check_differentiability(device, dtype, mglsadf, [(B, S), (B, S // P, M + 1)])


@pytest.mark.parametrize("phase", ["zero", "maximum"])
@pytest.mark.parametrize("ignore_gain", [False, True])
def test_zero_and_maximum_phase(
    device,
    dtype,
    phase,
    ignore_gain,
    alpha=0.42,
    M=24,
    P=80,
    L=400,
    fft_length=512,
    B=2,
):
    T = os.path.getsize("tools/SPTK/asset/data.short") // 2
    cmd_x = f"nrand -l {T}"
    x = torch.from_numpy(U.call(cmd_x)).to(
        device=device, dtype=torch.get_default_dtype() if dtype is None else dtype
    )

    cmd_mc = (
        f"x2x +sd tools/SPTK/asset/data.short | "
        f"frame -p {P} -l {L} | "
        f"window -w 1 -n 1 -l {L} -L {fft_length} | "
        f"mgcep -a {alpha} -m {M} -l {fft_length} -E -60"
    )
    mc = torch.from_numpy(U.call(cmd_mc).reshape(-1, M + 1)).to(
        device=device, dtype=torch.get_default_dtype() if dtype is None else dtype
    )

    common_params = {
        "filter_order": M,
        "frame_period": P,
        "ignore_gain": ignore_gain,
        "alpha": alpha,
        "phase": phase,
        "device": device,
        "dtype": dtype,
    }

    params1 = {"mode": "multi-stage", "cep_order": 200}
    mglsadf1 = diffsptk.MLSA(**common_params, **params1)
    y1 = mglsadf1(x, mc).cpu().numpy()

    params2 = {"mode": "single-stage", "ir_length": 200, "n_fft": 512}
    mglsadf2 = diffsptk.MLSA(**common_params, **params2)
    y2 = mglsadf2(x, mc).cpu().numpy()
    assert np.corrcoef(y1, y2)[0, 1] > 0.99

    params3 = {"mode": "freq-domain", "frame_length": L, "fft_length": fft_length}
    mglsadf3 = diffsptk.MLSA(**common_params, **params3)
    y3 = mglsadf3(x, mc).cpu().numpy()
    assert np.corrcoef(y1, y3)[0, 1] > 0.98

    S = T // 10
    U.check_differentiability(device, dtype, mglsadf1, [(B, S), (B, S // P, M + 1)])
    U.check_differentiability(device, dtype, mglsadf2, [(B, S), (B, S // P, M + 1)])
    U.check_differentiability(device, dtype, mglsadf3, [(B, S), (B, S // P, M + 1)])


@pytest.mark.parametrize("phase", ["zero", "maximum"])
@pytest.mark.parametrize("ignore_gain", [False, True])
def test_mixed_phase(
    device,
    dtype,
    phase,
    ignore_gain,
    alpha=0.42,
    M=24,
    P=80,
    L=400,
    fft_length=512,
    B=2,
):
    T = os.path.getsize("tools/SPTK/asset/data.short") // 2
    cmd_x = f"nrand -l {T}"
    x = torch.from_numpy(U.call(cmd_x)).to(
        device=device, dtype=torch.get_default_dtype() if dtype is None else dtype
    )

    cmd_mc = (
        f"x2x +sd tools/SPTK/asset/data.short | "
        f"frame -p {P} -l {L} | "
        f"window -w 1 -n 1 -l {L} -L {fft_length} | "
        f"mgcep -a {alpha} -m {M} -l {fft_length} -E -60"
    )
    mc = torch.from_numpy(U.call(cmd_mc).reshape(-1, M + 1)).to(
        device=device, dtype=torch.get_default_dtype() if dtype is None else dtype
    )
    if phase == "zero":
        half_mc = mc[..., 1:] * 0.5
        mc_mix = torch.cat([half_mc.flip(-1), mc[..., :1], half_mc], dim=-1)
    elif phase == "maximum":
        mc_mix = torch.cat([mc.flip(-1), 0 * mc[..., 1:]], dim=-1)

    common_params = {
        "filter_order": M,
        "frame_period": P,
        "ignore_gain": ignore_gain,
        "alpha": alpha,
        "device": device,
        "dtype": dtype,
    }

    params0 = {"mode": "multi-stage", "cep_order": 200}
    mglsadf0 = diffsptk.MLSA(**common_params, phase=phase, **params0)
    y0 = mglsadf0(x, mc).cpu().numpy()

    params1 = params0
    mglsadf1 = diffsptk.MLSA(**common_params, phase="mixed", **params1)
    y1 = mglsadf1(x, mc_mix).cpu().numpy()
    assert np.corrcoef(y1, y0)[0, 1] > 0.99

    params2 = {"mode": "single-stage", "ir_length": 200, "n_fft": 512}
    mglsadf2 = diffsptk.MLSA(**common_params, phase="mixed", **params2)
    y2 = mglsadf2(x, mc_mix).cpu().numpy()
    assert np.corrcoef(y1, y2)[0, 1] > 0.99

    params3 = {"mode": "freq-domain", "frame_length": L, "fft_length": fft_length}
    mglsadf3 = diffsptk.MLSA(**common_params, phase="mixed", **params3)
    y3 = mglsadf3(x, mc_mix).cpu().numpy()
    assert np.corrcoef(y1, y3)[0, 1] > 0.98

    S = T // 10
    U.check_differentiability(device, dtype, mglsadf1, [(B, S), (B, S // P, 2 * M + 1)])
    U.check_differentiability(device, dtype, mglsadf2, [(B, S), (B, S // P, 2 * M + 1)])
    U.check_differentiability(device, dtype, mglsadf3, [(B, S), (B, S // P, 2 * M + 1)])


@pytest.mark.parametrize("pade_order", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("chunk_length", [None, 1000])
def test_pade_approx(
    device, dtype, pade_order, chunk_length, alpha=0.42, M=24, P=80, L1=400, L2=512
):
    params = {"pade_order": pade_order, "cep_order": 100, "chunk_length": chunk_length}

    mglsadf = diffsptk.MLSA(
        M,
        P,
        alpha=alpha,
        mode="pade-approx",
        device=device,
        dtype=dtype,
        **params,
    )

    tmp1 = "mglsadf.tmp1"
    tmp2 = "mglsadf.tmp2"
    T = os.path.getsize("tools/SPTK/asset/data.short") // 2
    cmd1 = f"nrand -l {T} > {tmp1}"
    cmd2 = (
        f"x2x +sd tools/SPTK/asset/data.short | "
        f"frame -p {P} -l {L1} | "
        f"window -w 1 -n 1 -l {L1} -L {L2} | "
        f"mgcep -a {alpha} -m {M} -l {L2} -E -60 > {tmp2}"
    )
    U.check_compatibility(
        device,
        dtype,
        mglsadf,
        [cmd1, cmd2],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"mglsadf {tmp2} < {tmp1} -m {M} -p {P} -P {min(pade_order, 7)} -a {alpha}",
        [f"rm {tmp1} {tmp2}"],
        dx=[None, M + 1],
        eq=lambda a, b: np.corrcoef(a, b)[0, 1] > 0.98,
    )


@pytest.mark.parametrize("mode", ["multi-stage", "pade-approx"])
def test_learnable(mode, M=24, P=80, T=160):
    mglsadf = diffsptk.MLSA(M, P, mode=mode, learnable=True)
    U.check_learnable(mglsadf, [(T,), (T // P, M + 1)])
