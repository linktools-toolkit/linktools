#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Edge cases for the long-lived AI architecture invariants."""

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


def test_runtime_module_self_cycle_is_rejected(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {"a/module.py": "import linktools.ai.a.module\n"},
    )
    assert any(error.startswith("runtime module cycle:") for error in errors)


def test_cross_owner_static_attribute_obeys_all(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/__init__.py": "Public = object()\nHidden = object()\n__all__ = ['Public']\n",
            "b/use.py": "import linktools.ai.a as public_api\nvalue = public_api.Hidden\n",
        },
    )
    assert any("cross-owner non-exported symbol access" in error for error in errors)


def test_cross_owner_star_import_uses_static_all(tmp_path: Path) -> None:
    allowed = _errors(
        tmp_path / "allowed",
        {
            "a/__init__.py": "Public = object()\n__all__ = ['Public']\n",
            "b/use.py": "from linktools.ai.a import *\n",
        },
    )
    denied = _errors(
        tmp_path / "denied",
        {
            "a/public.py": "Public = object()\n",
            "b/use.py": "from linktools.ai.a.public import *\n",
        },
    )
    assert not allowed
    assert any("requires static __all__" in error for error in denied)


def test_conditional_all_is_not_a_static_export_contract(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/__init__.py": "Public = object()\nif True:\n    __all__ = ['Public']\n",
        },
    )
    assert any("__all__ must be one static string sequence" in error for error in errors)


def test_function_local_all_does_not_define_module_exports(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "a/__init__.py": (
                "def local():\n"
                "    __all__ = ['local-only']\n"
                "Public = object()\n"
                "__all__ = ['Public']\n"
            ),
        },
    )
    assert not errors


def test_pydantic_tool_control_owner_is_layout_independent(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "bridge/control.py": (
                "_OWNS_PYDANTIC_TOOL_CONTROL = True\n"
                "from pydantic_ai.exceptions import ModelRetry, ToolFailed\n"
            ),
        },
    )
    assert not any("Pydantic tool control" in error for error in errors)


def test_multiple_pydantic_tool_control_owners_are_rejected(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "first/control.py": (
                "_OWNS_PYDANTIC_TOOL_CONTROL = True\n"
                "from pydantic_ai.exceptions import ModelRetry\n"
            ),
            "second/control.py": (
                "_OWNS_PYDANTIC_TOOL_CONTROL = True\n"
                "from pydantic_ai.exceptions import ToolFailed\n"
            ),
        },
    )
    assert any("multiple production Pydantic tool control owners" in error for error in errors)


def test_pydantic_tool_controls_have_one_production_owner(tmp_path: Path) -> None:
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


def test_pydantic_tool_control_attribute_alias_is_rejected(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "bridge/control.py": (
                "_OWNS_PYDANTIC_TOOL_CONTROL = True\n"
                "from pydantic_ai.exceptions import ModelRetry\n"
            ),
            "capability/tool.py": (
                "import pydantic_ai.exceptions as exceptions\n"
                "value = exceptions.ModelRetry('retry')\n"
            ),
        },
    )
    assert any("direct Pydantic tool control access" in error for error in errors)


def test_pydantic_tool_control_module_import_is_checked_at_use(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "bridge/control.py": (
                "_OWNS_PYDANTIC_TOOL_CONTROL = True\n"
                "from pydantic_ai.exceptions import ToolFailed\n"
            ),
            "capability/tool.py": (
                "from pydantic_ai import exceptions as pydantic_exceptions\n"
                "value = pydantic_exceptions.ModelRetry('retry')\n"
            ),
        },
    )
    assert any("direct Pydantic tool control access" in error for error in errors)


def test_unrelated_pydantic_module_import_is_allowed(tmp_path: Path) -> None:
    errors = _errors(
        tmp_path,
        {
            "capability/tool.py": (
                "import pydantic_ai\n"
                "value = pydantic_ai.__name__\n"
            ),
        },
    )
    assert not any("Pydantic tool control" in error for error in errors)
