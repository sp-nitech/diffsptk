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

from typing import Any, Callable, TypeAlias, TypeVar

import numpy as np
import torch

Callable: TypeAlias = Callable[..., Any]

T = TypeVar("T", int, float)
ArrayLike: TypeAlias = list[T] | tuple[T, ...] | np.ndarray

Precomputed: TypeAlias = (
    tuple[
        tuple[Any, ...] | None,
        tuple[Callable, ...] | None,
        tuple[torch.Tensor, ...] | None,
    ]
    | tuple[Any, ...]
)
