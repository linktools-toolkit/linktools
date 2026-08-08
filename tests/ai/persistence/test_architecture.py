#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Persistence boundary checks for the converged Runtime architecture."""

import ast
from pathlib import Path


def test_runtime_has_no_provider_or_adapter_dependency() -> None:
    root = Path("linktools-ai/src/linktools/ai/runtime")
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.startswith("linktools.ai"):
                raise AssertionError(f"runtime uses absolute AI import: {path}:{node.lineno}")
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0] in {"adapter", "workspace", "temporal", "app"}:
                raise AssertionError(f"runtime imports forbidden package: {path}:{node.lineno}")


def test_converged_roots_and_owners_exist() -> None:
    root = Path("linktools-ai/src/linktools/ai")
    assert (root / "workspace").is_dir()
    assert (root / "app").is_dir()
    assert not (root / "local").exists()
    assert not (root / "entry").exists()
    assert not (root / "runtime" / "factory.py").exists()
    assert not (root / "runtime" / "container.py").exists()
    assert (root / "app" / "facade.py").is_file()
    assert (root / "app" / "assembly.py").is_file()
    assert (root / "app" / "workbench.py").is_file()
    assert (root / "app" / "bootstrap.py").is_file()
    assert (root / "adapter" / "repository.py").is_file()
    assert (root / "adapter" / "history.py").is_file()
    assert (root / "adapter" / "step.py").is_file()
    assert not (root / "adapter" / "file.py").exists()
