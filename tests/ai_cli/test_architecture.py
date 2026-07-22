#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture tests.

 -- ``commands/ai`` holds only the five frozen command files.
 -- command files import only ``linktools.ai_cli.*`` (+ stdlib/``linktools.cli``);
         they must NOT import Textual or Runtime internals (storage/runner/mcp/registry).
 -- the core ``linktools.ai`` package does not depend on Textual.

The spec writes the path as ``linktools-ai-cli/src/linktools/commands/ai``
(a notional separate package); this build keeps the CLI in place under
``linktools-ai`` (in-place refactor), so the real path is used here."""

import ast
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_COMMANDS_AI = _REPO / "linktools-ai" / "src" / "linktools" / "commands" / "ai"
_AI_PKG = _REPO / "linktools-ai" / "src" / "linktools" / "ai"

# Modules the command layer is forbidden to import directly. They
# are reached transitively through linktools.ai_cli, never from a command shell.
_FORBIDDEN_COMMAND_PREFIXES = (
    "textual",
    "linktools.ai.storage",
    "linktools.ai.agent.engine",
    "linktools.ai.mcp",
    "linktools.ai.registry",
)


def _imported_modules(source: str) -> "list[str]":
    """Flat list of fully-qualified module names a source file imports."""
    tree = ast.parse(source)
    mods: "list[str]" = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module)
    return mods


class TestCommandsAiContents(unittest.TestCase):
    """-- only the five command files (+ package marker) live here."""

    def test_only_command_files_present(self):
        expected = {
            "__init__.py",
            "tui.py",
            "init.py",
            "run.py",
            "continue_.py",
            "doctor.py",
        }
        actual = {
            p.name for p in _COMMANDS_AI.iterdir() if p.is_file() and p.suffix == ".py"
        }
        self.assertEqual(actual, expected)


class TestCommandsAiImports(unittest.TestCase):
    """-- command files import only linktools.ai_cli.* (+ stdlib/cli)."""

    def test_no_forbidden_imports_in_command_files(self):
        offenders: "list[str]" = []
        for path in sorted(_COMMANDS_AI.glob("*.py")):
            modules = _imported_modules(path.read_text(encoding="utf-8"))
            for module in modules:
                if any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in _FORBIDDEN_COMMAND_PREFIXES
                ):
                    offenders.append(f"{path.name}: {module}")
        self.assertFalse(offenders, f"forbidden command imports:\n{offenders}")

    def test_commands_route_through_ai_cli(self):
        # Every non-init command file imports at least one linktools.ai_cli.* name.
        for path in sorted(_COMMANDS_AI.glob("*.py")):
            if path.name == "__init__.py":
                continue
            modules = _imported_modules(path.read_text(encoding="utf-8"))
            self.assertTrue(
                any(
                    module == "linktools.ai_cli"
                    or module.startswith("linktools.ai_cli.")
                    for module in modules
                ),
                f"{path.name} does not route through linktools.ai_cli",
            )


class TestCoreAuiHasNoTextual(unittest.TestCase):
    """-- linktools.ai never imports Textual."""

    def test_no_textual_import_in_core_ai(self):
        offenders: "list[str]" = []
        for path in sorted(_AI_PKG.rglob("*.py")):
            modules = _imported_modules(path.read_text(encoding="utf-8"))
            for module in modules:
                if module == "textual" or module.startswith("textual."):
                    offenders.append(f"{path.relative_to(_AI_PKG)}: {module}")
        self.assertFalse(offenders, f"textual leaked into linktools.ai:\n{offenders}")


if __name__ == "__main__":
    unittest.main()
