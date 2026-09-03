#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from scripts.check.ai.architecture import ArchitecturePolicyChecker


def _source_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "linktools" / "ai"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name != "__init__.py":
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


def test_runtime_module_cycle_is_rejected(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/x.py": "from . import y\n",
            "a/y.py": "from . import x\n",
        },
    )
    assert any(error.startswith("runtime module cycle:") for error in errors)


def test_runtime_owner_cycle_is_rejected(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/__init__.py": "from .public import A\n__all__ = ['A']\n",
            "a/public.py": "from linktools.ai.b import B\nA = object()\n__all__ = ['A']\n",
            "b/__init__.py": "from .public import B\n__all__ = ['B']\n",
            "b/public.py": "from linktools.ai.a import A\nB = object()\n__all__ = ['B']\n",
        },
    )
    assert any(error.startswith("runtime owner cycle:") for error in errors)


def test_type_checking_back_reference_does_not_create_runtime_cycle(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/__init__.py": "from .public import A\n__all__ = ['A']\n",
            "a/public.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from linktools.ai.b import B\nA = object()\n__all__ = ['A']\n",
            "b/__init__.py": "from .public import B\n__all__ = ['B']\n",
            "b/public.py": "from linktools.ai.a import A\nB = object()\n__all__ = ['B']\n",
        },
    )
    assert not any("cycle:" in error for error in errors)


def test_cross_owner_private_module_is_rejected(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/_impl.py": "Public = object()\n__all__ = ['Public']\n",
            "b/use.py": "from linktools.ai.a._impl import Public\n",
        },
    )
    assert any("cross-owner private module access" in error for error in errors)


def test_cross_owner_non_exported_symbol_is_rejected(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/__init__.py": "Public = object()\nHidden = object()\n__all__ = ['Public']\n",
            "b/use.py": "from linktools.ai.a import Hidden\n",
        },
    )
    assert any("cross-owner non-exported symbol access" in error for error in errors)


def test_module_without_all_has_no_cross_owner_exports(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/public.py": "Public = object()\n",
            "b/use.py": "from linktools.ai.a.public import Public\n",
        },
    )
    assert any("cross-owner non-exported symbol access" in error for error in errors)


def test_same_owner_private_import_is_allowed(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/_impl.py": "value = object()\n",
            "a/public.py": "from ._impl import value\n__all__ = ['value']\n",
        },
    )
    assert not errors


def test_public_forwarding_facade_is_allowed(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/__init__.py": "from ._impl import Public\n__all__ = ['Public']\n",
            "a/_impl.py": "Public = object()\n",
            "b/use.py": "from linktools.ai.a import Public\n",
        },
    )
    assert not errors


def test_names_and_nested_packages_do_not_define_architecture(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "helpers.py": "value = 1\n",
            "utils.py": "value = 1\n",
            "manager.py": "value = 1\n",
            "common.py": "value = 1\n",
            "nested/deeper/module.py": "value = 1\n",
            "a/normalize.py": "def normalize(value):\n    return value\n",
            "b/normalize.py": "def normalize(value):\n    return value\n",
        },
    )
    assert not errors


def test_invalid_static_all_is_rejected(tmp_path: Path) -> None:
    duplicate = _errors(
        tmp_path / "duplicate",
        {"a/__init__.py": "Public = object()\n__all__ = ['Public', 'Public']\n"},
    )
    missing = _errors(
        tmp_path / "missing",
        {"a/__init__.py": "__all__ = ['Missing']\n"},
    )
    dynamic = _errors(
        tmp_path / "dynamic",
        {"a/__init__.py": "Public = object()\n__all__ = ['Public']\n__all__ += ['Other']\n"},
    )
    assert any("duplicate __all__ exports" in error for error in duplicate)
    assert any("__all__ exports unbound names" in error for error in missing)
    assert any("__all__ must be one static string sequence" in error for error in dynamic)


def test_external_production_consumer_uses_same_public_boundary(tmp_path: Path) -> None:
    source_root = _source_tree(
        tmp_path,
        {"a/__init__.py": "Public = object()\nHidden = object()\n__all__ = ['Public']\n"},
    )
    external = tmp_path / "external" / "linktools"
    consumer = external / "commands" / "ai.py"
    consumer.parent.mkdir(parents=True)
    consumer.write_text("from linktools.ai.a import Hidden\n", encoding="utf-8")
    errors = ArchitecturePolicyChecker().check(source_root, external_roots=(external,)).errors
    assert any("linktools.commands.ai: cross-owner non-exported symbol access" in error for error in errors)


def test_linktools_ai_top_level_exports_are_exact() -> None:
    import linktools.ai as ai

    assert ai.__all__ == [
        "Agent",
        "CapabilityGroup",
        "Execution",
        "RunContext",
        "Runtime",
        "Session",
        "Workspace",
    ]
