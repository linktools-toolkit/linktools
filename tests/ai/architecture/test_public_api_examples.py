#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public API surface: the entry points the README documents are importable
from their public paths, and the retired names are absent from the public
surface (importing a deleted module must fail, not silently succeed)."""

import importlib

import pytest

# Each public entry point the README references must resolve at its documented
# public path. (module_path, [names])
PUBLIC_IMPORTS = [
    ("linktools.ai.runtime", ["Runtime", "build_runtime"]),
    ("linktools.ai.storage", ["Storage", "FilesystemStorage"]),
    ("linktools.ai.asset", ["AssetStore"]),
    ("linktools.ai.artifact", ["ArtifactStore"]),
    ("linktools.ai.model", ["ModelResolver", "ModelPolicy"]),
    ("linktools.ai.sandbox", ["Sandbox"]),
]


@pytest.mark.parametrize("module_path,names", PUBLIC_IMPORTS)
def test_public_entry_point_is_importable(module_path: str, names: "list[str]") -> None:
    module = importlib.import_module(module_path)
    missing = [n for n in names if not hasattr(module, n)]
    assert not missing, f"{module_path} is missing public names: {missing}"


# Deleted modules / classes. Importing the module must fail, OR the name must
# be absent from every public package it was ever re-exported from.
DELETED_MODULES = [
    "linktools.ai.model.router",
    "linktools.ai.memory.index",
    "linktools.ai.memory.outbox",
]


@pytest.mark.parametrize("module_path", DELETED_MODULES)
def test_deleted_module_is_not_importable(module_path: str) -> None:
    with pytest.raises(ImportError):
        importlib.import_module(module_path)


DELETED_PUBLIC_NAMES = [
    ("linktools.ai.model", "ModelGateway"),
    ("linktools.ai.model", "ModelRouter"),
    ("linktools.ai.runtime", "ModelGateway"),
    ("linktools.ai.runtime", "ModelRouter"),
]


@pytest.mark.parametrize("module_path,name", DELETED_PUBLIC_NAMES)
def test_retired_name_absent_from_public_surface(module_path: str, name: str) -> None:
    module = importlib.import_module(module_path)
    assert not hasattr(module, name), (
        f"{name} is still re-exported from {module_path}; a deleted type must "
        "not remain on the public surface"
    )
