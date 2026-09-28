#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Representative checks for long-lived AI architecture invariants."""

import os
import subprocess
import sys
from pathlib import Path

from scripts.check.ai.architecture import ArchitecturePolicyChecker


def _source_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "linktools" / "ai"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        package = path.parent
        while package != root:
            init = package / "__init__.py"
            if not init.exists():
                init.write_text("__all__ = []\n", encoding="utf-8")
            package = package.parent
        path.write_text(source, encoding="utf-8")
    return root


def _errors(tmp_path: Path, files: dict[str, str]) -> tuple[str, ...]:
    return ArchitecturePolicyChecker().check(_source_tree(tmp_path, files)).errors


def test_runtime_cycles_are_rejected(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/x.py": "from . import y\n",
            "a/y.py": "from . import x\n",
        },
    )
    assert any(error.startswith("runtime module cycle:") for error in errors)


def test_type_checking_back_reference_is_not_a_runtime_cycle(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/__init__.py": "from .public import A\n__all__ = ['A']\n",
            "a/public.py": (
                "from typing import TYPE_CHECKING\n"
                "if TYPE_CHECKING:\n"
                "    from linktools.ai.b import B\n"
                "A = object()\n"
                "__all__ = ['A']\n"
            ),
            "b/__init__.py": "from .public import B\n__all__ = ['B']\n",
            "b/public.py": "from linktools.ai.a import A\nB = object()\n__all__ = ['B']\n",
        },
    )
    assert not any("cycle:" in error for error in errors)


def test_cross_owner_private_and_non_exported_access_are_rejected(tmp_path: Path) -> None:
    private_errors = _errors(
        tmp_path / "private",
        {
            "a/_impl.py": "Public = object()\n__all__ = ['Public']\n",
            "b/use.py": "from linktools.ai.a._impl import Public\n",
        },
    )
    export_errors = _errors(
        tmp_path / "export",
        {
            "a/__init__.py": "Public = object()\nHidden = object()\n__all__ = ['Public']\n",
            "b/use.py": "from linktools.ai.a import Hidden\n",
        },
    )
    assert any("cross-owner private module access" in error for error in private_errors)
    assert any("cross-owner non-exported symbol access" in error for error in export_errors)


def test_same_owner_private_implementation_can_back_public_surface(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/__init__.py": "from ._impl import Public\n__all__ = ['Public']\n",
            "a/_impl.py": "Public = object()\n",
            "b/use.py": "from linktools.ai.a import Public\n",
        },
    )
    assert not errors


def test_exports_must_be_declared_statically(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/__init__.py": (
                "Public = object()\n"
                "__all__ = ['Public']\n"
                "__all__ += ['Other']\n"
            ),
        },
    )
    assert any("__all__ must be one static string sequence" in error for error in errors)


def test_external_production_consumers_use_the_same_public_boundary(tmp_path: Path) -> None:
    source_root = _source_tree(
        tmp_path,
        {"a/__init__.py": "Public = object()\nHidden = object()\n__all__ = ['Public']\n"},
    )
    external = tmp_path / "external" / "linktools"
    consumer = external / "commands" / "ai.py"
    consumer.parent.mkdir(parents=True)
    consumer.write_text("from linktools.ai.a import Hidden\n", encoding="utf-8")

    errors = ArchitecturePolicyChecker().check(
        source_root,
        external_roots=(external,),
    ).errors

    assert any("cross-owner non-exported symbol access" in error for error in errors)


def test_pydantic_tool_control_has_one_declared_owner(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "bridge/control.py": (
                "_OWNS_PYDANTIC_TOOL_CONTROL = True\n"
                "from pydantic_ai.exceptions import ModelRetry, ToolFailed\n"
            ),
            "capability/tool.py": "from pydantic_ai.exceptions import ModelRetry\n",
        },
    )
    assert any("direct Pydantic tool control access" in error for error in errors)


def test_optional_dependencies_do_not_leak_into_asset_import() -> None:
    environment = dict(os.environ)
    source_root = Path(__file__).parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root / "linktools-ai/src"), str(source_root / "linktools/src"))
    )
    blocker = """
import importlib
import sys
from importlib.abc import MetaPathFinder

class Blocker(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in TARGETS):
            raise ModuleNotFoundError(fullname, name=fullname.split('.')[0])
        return None

TARGETS = ('sqlalchemy', 'acp')
for target in TARGETS:
    sys.meta_path.insert(0, Blocker())
importlib.import_module('linktools.ai.asset')
for name in TARGETS:
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", blocker], env=environment, check=True)
