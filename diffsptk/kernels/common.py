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

import ctypes
import warnings
from importlib import resources
from typing import Any

import torch

_C_TYPES = {torch.float32: "float", torch.float64: "double"}
_KERNELS: dict[tuple, Any] = {}
_CONTEXTS: dict[int, ctypes.c_void_p] = {}
_disabled = False


class KernelUnavailableError(RuntimeError):
    """Raised when a kernel cannot be compiled or loaded."""


def is_available() -> bool:
    """Return True if CUDA kernels can be compiled at runtime."""
    return (
        not _disabled
        and torch.cuda.is_available()
        and not torch.version.hip  # type: ignore[attr-defined]
        and hasattr(torch.cuda, "_compile_kernel")
    )


def is_supported(*tensors: torch.Tensor) -> bool:
    """Return True if the given tensors can be processed by the CUDA kernels."""
    if not is_available():
        return False
    t0 = tensors[0]
    if t0.dtype not in _C_TYPES:
        return False
    return all(
        t.is_cuda and t.device == t0.device and t.dtype == t0.dtype for t in tensors
    )


def _check(result: int) -> None:
    if result != 0:
        raise RuntimeError(f"CUDA driver API call failed with error code {result}.")


def _driver() -> Any:
    from torch.cuda import _utils

    return _utils._get_gpu_runtime_library()  # type: ignore[attr-defined]


def _ensure_context(device: torch.device) -> ctypes.c_void_p | None:
    """Make the primary context of the device current on the calling thread.

    The kernels are loaded and launched via the CUDA driver API, which requires a
    current context. Threads that have not called the CUDA runtime API yet, e.g.,
    the worker thread of the autograd engine, may not have it.

    Returns the previously current context if it has been replaced, else None.
    """
    lib = _driver()
    ctx = ctypes.c_void_p()
    _check(lib.cuCtxGetCurrent(ctypes.byref(ctx)))
    if ctx.value:
        dev = ctypes.c_int()
        _check(lib.cuCtxGetDevice(ctypes.byref(dev)))
        if dev.value == device.index:
            return None

    if device.index not in _CONTEXTS:
        dev = ctypes.c_int()
        _check(lib.cuDeviceGet(ctypes.byref(dev), device.index))
        primary = ctypes.c_void_p()
        _check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(primary), dev))
        _CONTEXTS[device.index] = primary
    _check(lib.cuCtxSetCurrent(_CONTEXTS[device.index]))
    return ctx


def _restore_context(ctx: ctypes.c_void_p | None) -> None:
    if ctx is not None:
        _check(_driver().cuCtxSetCurrent(ctx))


def get_kernel(
    source: str,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    defines: tuple[tuple[str, int], ...] = (),
) -> Any:
    """Compile (or fetch from the cache) a kernel defined in a .cu file.

    If the compilation fails, the kernels are disabled for the rest of the process
    and KernelUnavailableError is raised so that the caller can fall back to a
    PyTorch implementation.
    """
    global _disabled
    key = (source, name, dtype, device.index, defines)
    if key not in _KERNELS:
        code = resources.files("diffsptk.kernels").joinpath(source).read_text()
        options = [f"-DSCALAR_T={_C_TYPES[dtype]}"]
        options += [f"-D{k}={v}" for k, v in defines]
        try:
            with torch.cuda.device(device):
                _KERNELS[key] = torch.cuda._compile_kernel(  # type: ignore[attr-defined]
                    code, name, nvcc_options=options
                )
        except Exception as e:
            _disabled = True
            warnings.warn(
                f"Failed to compile the CUDA kernel {name!r}; "
                f"falling back to the PyTorch implementation: {e}",
                stacklevel=2,
            )
            raise KernelUnavailableError(str(e)) from e
    return _KERNELS[key]


def prepare(
    source: str,
    names: tuple[str, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    defines: tuple[tuple[str, int], ...] = (),
) -> None:
    """Compile the kernels in advance to detect a compilation failure early."""
    previous = _ensure_context(device)
    try:
        for name in names:
            get_kernel(source, name, dtype, device, defines)
    finally:
        _restore_context(previous)


def launch(
    source: str,
    name: str,
    n_threads: int,
    args: list,
    *,
    dtype: torch.dtype,
    device: torch.device,
    defines: tuple[tuple[str, int], ...] = (),
    block_size: int = 256,
) -> None:
    """Launch a 1-D kernel with at least n_threads threads."""
    if n_threads <= 0:
        return
    n_blocks = (n_threads + block_size - 1) // block_size
    previous = _ensure_context(device)
    try:
        kernel = get_kernel(source, name, dtype, device, defines)
        kernel(grid=(n_blocks, 1, 1), block=(block_size, 1, 1), args=args)
    finally:
        _restore_context(previous)
