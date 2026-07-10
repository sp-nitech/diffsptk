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

import ast
import inspect
import re
from pathlib import Path

import pytest
from torch import Tensor

from diffsptk import functional, modules

FUNCTIONAL_PATH = Path(functional.__file__)
MODULES_DIR = Path(modules.__file__).parent

DELEGATION_EXCEPTIONS = {"dtw_merge": "DynamicTimeWarping"}


def _delegation_map() -> dict[str, str]:
    """Map each functional function name to the module class it delegates to."""
    tree = ast.parse(FUNCTIONAL_PATH.read_text())
    mapping: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            f = call.func
            # Match `nn.<Class>.<method>(...)`.
            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Attribute)
                and isinstance(f.value.value, ast.Name)
                and f.value.value.id == "nn"
            ):
                mapping[node.name] = f.value.attr
    return mapping


DELEGATION = _delegation_map()


def _functional_classes() -> list[tuple[str, ast.ClassDef]]:
    """Yield (qualified name, class node) for every BaseFunctionalModule."""
    result = []
    for path in sorted(MODULES_DIR.glob("*.py")):
        if path.name.startswith(("#", "__")):
            continue
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and any(
                isinstance(b, ast.Name) and b.id == "BaseFunctionalModule"
                for b in node.bases
            ):
                result.append((f"{path.stem}.{node.name}", node))
    return result


FUNCTIONAL_CLASSES = _functional_classes()


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _precompute_dict_keys(cls: ast.ClassDef) -> set[str]:
    """Collect the keys of every dict-style Precomputed field in _precompute."""
    prec = _method(cls, "_precompute")
    keys: set[str] = set()
    if prec is None:
        return keys
    for node in ast.walk(prec):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Precomputed"
        ):
            for kw in node.keywords:
                if kw.arg in ("values", "layers", "tensors") and isinstance(
                    kw.value, ast.Dict
                ):
                    for key in kw.value.keys:
                        if isinstance(key, ast.Constant):
                            keys.add(key.value)
    return keys


def _doc_params(doc: str | None) -> set[str]:
    """Extract parameter names from a numpydoc-style Parameters section."""
    if not doc:
        return set()
    names: set[str] = set()
    in_params = False
    for line in doc.splitlines():
        s = line.strip()
        if s == "Parameters":
            in_params = True
            continue
        if in_params and s in ("Returns", "Examples", "References", "Notes"):
            break
        if not in_params:
            continue
        m = re.match(r"^(\w[\w, ]*) :", s)
        if m:
            for name in m.group(1).split(","):
                names.add(name.strip())
    return names


def _class_params(cls: type) -> set[str]:
    params = _doc_params(cls.__doc__)
    forward = getattr(cls, "forward", None)
    if forward is not None:
        params |= _doc_params(forward.__doc__)
    try:
        params |= set(inspect.signature(cls.__init__).parameters)
    except (ValueError, TypeError):
        pass
    return params - {"self", "args", "kwargs"}


def test_all_functional_functions_are_covered():
    public = {
        name
        for name in dir(functional)
        if not name.startswith("_") and callable(getattr(functional, name))
    }
    uncovered = public - set(DELEGATION) - set(DELEGATION_EXCEPTIONS)
    uncovered = {
        n for n in uncovered if getattr(functional, n).__module__ == functional.__name__
    }
    assert not uncovered, f"functional functions without a known contract: {uncovered}"


@pytest.mark.parametrize("func_name", sorted(DELEGATION))
def test_functional_delegates_to_existing_module(func_name):
    cls_name = DELEGATION[func_name]
    assert hasattr(modules, cls_name), (
        f"functional.{func_name} delegates to unknown module class {cls_name}"
    )


@pytest.mark.parametrize(
    "func_name", sorted(set(DELEGATION) - set(DELEGATION_EXCEPTIONS))
)
def test_functional_params_are_documented_by_module(func_name):
    cls_name = DELEGATION[func_name]
    cls = getattr(modules, cls_name)
    func = getattr(functional, func_name)

    func_params = {
        name
        for name, p in inspect.signature(func).parameters.items()
        if name not in ("args", "kwargs") and p.annotation not in (Tensor, "Tensor")
    }
    class_params = _class_params(cls)

    missing = func_params - class_params
    assert not missing, (
        f"functional.{func_name} exposes {sorted(missing)} but {cls_name} "
        f"documents {sorted(class_params)}"
    )


@pytest.mark.parametrize(
    "qualname,cls", FUNCTIONAL_CLASSES, ids=[q for q, _ in FUNCTIONAL_CLASSES]
)
def test_forward_names_match_precompute(qualname, cls):
    forward = _method(cls, "_forward")
    if forward is None or not forward.args.kwonlyargs:
        return

    n_kwonly = len(forward.args.kwonlyargs)
    required = {
        a.arg
        for a, default in zip(forward.args.kwonlyargs, forward.args.kw_defaults)
        if default is None
    }
    assert len(forward.args.kw_defaults) == n_kwonly  # sanity

    produced = _precompute_dict_keys(cls)
    if not produced:
        return

    missing = required - produced
    assert not missing, (
        f"{qualname}._forward requires {sorted(missing)} but _precompute "
        f"produces {sorted(produced)}"
    )
