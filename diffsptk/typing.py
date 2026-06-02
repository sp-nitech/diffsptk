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

from typing import Any, NamedTuple, TypeAlias, TypeVar
from typing import Callable as _Callable

import numpy as np
import torch

Callable: TypeAlias = _Callable[..., Any]

T = TypeVar("T", int, float)
ArrayLike: TypeAlias = (
    list[T] | tuple[T, ...] | list[list[T]] | tuple[tuple[T, ...], ...] | np.ndarray
)


class Precomputed(NamedTuple):
    values: tuple[Any, ...] = ()
    layers: tuple[Callable, ...] = ()
    tensors: tuple[torch.Tensor, ...] = ()
