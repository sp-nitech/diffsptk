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

from abc import ABC, abstractmethod
from typing import Any, ClassVar, NamedTuple

import torch
from torch import nn

from ..typing import Callable


class Precomputed(NamedTuple):
    """Precomputed components derived from the given parameters."""

    values: dict[str, Any] = {}
    layers: dict[str, Callable] = {}
    tensors: dict[str, torch.Tensor] = {}


class BaseNonFunctionalModule(ABC, nn.Module):
    """Marker class for modules that provide no functional interface."""


class BaseFunctionalModule(ABC, nn.Module):
    """Base class for modules that provide a functional interface."""

    # Whether the first parameter of _precompute is the input size.
    _takes_input_size: ClassVar[bool] = False

    # Names of the values/layers passed to _forward by keyword.
    _value_names: tuple[str, ...] = ()
    _layer_names: tuple[str, ...] = ()

    def _register_precomputed(
        self,
        precomputed: Precomputed,
        learnable: bool = False,
    ) -> None:
        """Store the precomputed values, layers, and tensors in this module."""

        self._value_names = tuple(precomputed.values)
        for name, value in precomputed.values.items():
            setattr(self, name, value)

        self._layer_names = tuple(precomputed.layers)
        for name, layer in precomputed.layers.items():
            setattr(self, name, layer)

        for name, tensor in precomputed.tensors.items():
            if learnable:
                setattr(self, name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor, persistent=False)

    def _call_forward(self, *args) -> Any:
        """Call _forward with the stored state resolved by name."""
        named = {name: getattr(self, name) for name in self._value_names}
        named.update({name: getattr(self, name) for name in self._layer_names})
        named.update(self._buffers)
        named.update(self._parameters)
        return self._forward(*args, **named)

    @classmethod
    def _apply_precomputed(cls, precomputed: Precomputed, **inputs: Any) -> Any:
        """Call _forward with a name-binding Precomputed."""
        state = {**precomputed.values, **precomputed.layers, **precomputed.tensors}
        return cls._forward(**inputs, **state)

    @staticmethod
    @abstractmethod
    def _func(*args, **kwargs) -> Any:
        """Perform the operation in a functional manner."""

    @staticmethod
    @abstractmethod
    def _check(*args, **kwargs) -> None:
        """Validate the given parameters."""

    @staticmethod
    @abstractmethod
    def _precompute(*args, **kwargs) -> Precomputed:
        """Precompute the values, layers, and tensors used in _forward."""

    @staticmethod
    @abstractmethod
    def _forward(*args, **kwargs) -> Any:
        """Perform the core operation."""


class BaseLearnerModule(ABC, nn.Module):
    """Base class for modules that learn their parameters from data."""

    @abstractmethod
    def transform(self, *args, **kwargs) -> Any:
        """Transform the input using the learned parameters."""
