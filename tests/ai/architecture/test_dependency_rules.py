#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dependency-direction guards.

A "rule" maps a top-level package (the importer, under
``linktools/ai/<pkg>/``) to the set of other top-level packages it must not
import. The check is AST-based: only ``import x.y`` and ``from x.y import z``
statements are considered (including relative imports, resolved against the
importing file's location), and the importer's own sub-tree never counts as a
self-dependency.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_AI = _REPO / "linktools-ai" / "src" / "linktools" / "ai"

# Importer top-level package -> set of forbidden top-level package names.
FORBIDDEN_IMPORTS: "dict[str, set[str]]" = {
    # storage is the bottom-layer dependency: it must not reach up into any
    # business domain. A backend is injected INTO storage; storage never
    # imports those layers.
    "storage": {
        "agent", "artifact", "execution", "governance", "model",
        "observability", "runtime", "spec", "tasks", "tool",
    },
    # spec does not import runtime, execution, or agent.
    "spec": {"runtime", "execution", "agent"},
    # runtime/facade only imports ExecutionService, ExecutionQueryService, and
    # public DTOs -- never a concrete Store or the compiler/engine directly.
    # Enforced precisely (not just at package granularity) by
    # test_architecture_section_7_6; the package-level rule here additionally
    # guards that agent/execution never import runtime at all -- only runtime
    # may reach into every composition dependency.
    "agent": {"runtime"},
    "execution": {"runtime"},
    "tool": {"runtime"},
    "tasks": {"runtime"},
    "governance": {"runtime"},
}


def _top_level_packages() -> "set[str]":
    return {
        p.name
        for p in _AI.iterdir()
        if p.is_dir() and p.name != "__pycache__"
    }


def _imports_in(file_path: Path) -> "set[str]":
    """Top-level ``linktools.ai.<x>`` packages imported by this file.

    Resolves relative imports (``from ..execution.domain import X``) against
    the file's location, so a forbidden dep cannot slip in via a dotted-
    relative form that an absolute-import-only check would miss.
    """
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return set()
    rel = file_path.relative_to(_AI).with_suffix("")
    is_init = rel.name == "__init__"
    if is_init:
        rel = rel.parent
    parts = ("linktools", "ai") + tuple(rel.parts)
    base_pkg = ".".join(parts) if is_init else ".".join(parts[:-1])

    roots: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    roots.add(node.module)
            else:
                base = base_pkg
                for _ in range(node.level - 1):
                    if "." in base:
                        base = base.rsplit(".", 1)[0]
                    else:
                        base = ""
                        break
                if node.module:
                    roots.add(f"{base}.{node.module}")
                else:
                    for alias in node.names:
                        roots.add(f"{base}.{alias.name}")
    out: "set[str]" = set()
    for name in roots:
        if name.startswith("linktools.ai.") or name == "linktools.ai":
            tail = name.split(".")[2] if name.startswith("linktools.ai.") else ""
            if tail:
                out.add(tail)
    return out


def _package_imports(pkg: str) -> "dict[Path, set[str]]":
    pkg_dir = _AI / pkg
    if not pkg_dir.is_dir():
        return {}
    result: "dict[Path, set[str]]" = {}
    for path in pkg_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        result[path] = _imports_in(path)
    return result


@pytest.mark.parametrize("importer", sorted(FORBIDDEN_IMPORTS))
def test_forbidden_dependency_directions_hold(importer: str) -> None:
    """Enforced dependency rules must not be violated by new code."""
    forbidden = FORBIDDEN_IMPORTS[importer]
    if not (_AI / importer).is_dir():
        pytest.skip(f"package {importer!r} not present in this checkout")
    violations: "list[str]" = []
    for path, imported in _package_imports(importer).items():
        bad = imported & forbidden
        if bad:
            rel = path.relative_to(_REPO)
            violations.append(f"{rel}: imports {sorted(bad)}")
    assert not violations, (
        f"{importer}/ imports a forbidden package:\n  " + "\n  ".join(violations)
    )


def _two_cycles() -> "set[tuple[str, str]]":
    """All 2-cycles (A imports B AND B imports A) between top-level packages."""
    pkgs = _top_level_packages()
    edges: "dict[str, set[str]]" = {p: set() for p in pkgs}
    for pkg in pkgs:
        imported: "set[str]" = set()
        for imps in _package_imports(pkg).values():
            imported |= imps
        edges[pkg] = imported & pkgs
    cycles: "set[tuple[str, str]]" = set()
    for a in pkgs:
        for b in edges[a]:
            if a != b and a in edges.get(b, set()):
                cycles.add(tuple(sorted((a, b))))  # type: ignore[arg-type]
    return cycles


# Top-level 2-cycles present today, each a deliberate cross-wiring rather than
# debt:
#
# - (agent, execution): execution/service.py compiles specs via
#   agent.compiler/agent.engine (the execution service is explicitly permitted
#   to depend on the agent compiler/engine); agent/models.py and
#   agent/engine.py reference execution.domain's RunErrorInfo/RunResult and
#   execution.context.RunContext as the agent-outcome/context boundary types.
# - (agent, governance): agent's security/sandbox capability wiring reaches
#   governance's policy/security types; governance's only reference back is a
#   TYPE_CHECKING-only annotation (agent.capability.exposure), not a runtime
#   dependency.
# - (agent, tool): agent compiles tool policy capabilities from tool.pydantic;
#   tool's builtin/sandbox toolsets reach agent's dependency/context types.
# - (governance, tool): governance's command/path policy governs tool
#   execution; tool's security pipeline wiring reaches governance types.
#
# New cycles beyond this set must not appear.
_BASELINE_TWO_CYCLES: "frozenset[tuple[str, str]]" = frozenset({
    ("agent", "execution"),
    ("agent", "governance"),
    ("agent", "tool"),
    ("governance", "tool"),
})


def test_no_new_circular_top_level_imports() -> None:
    """No NEW 2-cycle may appear between top-level packages.

    The baseline carries a known, deliberate set of 2-cycles (recorded above).
    This asserts the current set never grows beyond the baseline.
    """
    current = _two_cycles()
    new = current - _BASELINE_TWO_CYCLES
    assert not new, (
        f"new top-level 2-cycles introduced (refactor must not add cycles): "
        f"{sorted(new)}"
    )


def test_baseline_cycles_are_still_accurate() -> None:
    """The recorded baseline must not silently over-allow: every entry must
    still be a real cycle, so a resolved cycle is promptly removed from the
    allowlist instead of becoming permanent dead weight."""
    current = _two_cycles()
    stale = _BASELINE_TWO_CYCLES - current
    assert not stale, (
        f"baseline lists cycles that no longer exist -- remove them: {sorted(stale)}"
    )
