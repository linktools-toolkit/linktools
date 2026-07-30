#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structural conformance gates for the d1b4c91 storage/runtime spec.

Encodes the directory invariants from
``.docs/linktools_ai_execution_runtime_storage_final_spec_d1b4c91_corrected.md``
sections 2.1, 2.32 and 4.8: the forbidden top-level packages must not exist,
the execution domain types must be defined exactly once, the top-level layout
must match the final converged set, and the deletion manifest must be honored.

This is a ratchet: end-state assertions ``xfail`` until their phase lands, so
the branch stays green while the target is tracked explicitly. As each phase
completes, add its key to :data:`LANDED` and the assertion becomes a hard
gate.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_AI = _REPO / "linktools-ai" / "src" / "linktools" / "ai"

# Phases of the d1b4c91 spec that have fully landed. Add a key here as the
# corresponding structural change completes; the matching gate then flips from
# ``xfail`` to a hard pass/fail gate.
LANDED: set[str] = {
    "forbidden_pkgs",
    "closed_layout",
    "agent_usage_unified",
    "deletion_manifest",
}

# Spec 2.1 / 2.32: these top-level packages must never (re)appear.
FORBIDDEN_TOP_LEVEL_PACKAGES = (
    "run",
    "catalog",
    "resources",
    "events",
)

# Spec 2.1: the final converged top-level package set.
EXPECTED_TOP_LEVEL_PACKAGES = {
    "agent",
    "artifact",
    "evaluation",
    "execution",
    "governance",
    "model",
    "observability",
    "runtime",
    "spec",
    "storage",
    "tasks",
    "tool",
}

# Spec 2.3 / 3.2: each execution domain type is defined exactly once.
SINGLE_DEFINITION_TYPES = (
    "RunStatus",
    "RunKind",
    "RunRecord",
    "RunDefinition",
    "RunApproval",
    "RunUsage",
)

# Spec 2.1 / 2.32 deletion manifest. Each must be removed once its live code is
# migrated to the named owner. NOTE two deliberate user overrides:
# - execution/persistence/local.py and agent/memory/persistence/filesystem.py
#   are RETAINED as local/debug backends (the file-based stores stay available
#   for debugging and test scenarios even though SQL is the production
#   default).
# - execution/store.py is RETAINED under its established name: every domain
#   (artifact/spec/tasks/tool/execution) uses the same `{Domain}Backend`
#   Protocol + `{Domain}Store` wrapper-with-`.backend` shape in a file named
#   `store.py`. Renaming only execution's to `ports.py` would break that
#   global naming consistency, so the "delete the wrapper, delete .backend"
#   directive is overridden repo-wide by this established pattern.
DELETED_MODULES = (
    "runtime/executor.py",
    "execution/run.py",
    "execution/models.py",
    "execution/commit.py",
    "storage/json.py",
)


def _top_level_packages() -> set[str]:
    return {
        p.name
        for p in _AI.iterdir()
        if p.is_dir() and p.name != "__pycache__"
    }


def _require(phase: str):
    return pytest.mark.xfail(
        reason=f"d1b4c91 phase {phase!r} not yet landed",
        run=False,
        strict=True,
    )


@pytest.mark.skipif("forbidden_pkgs" not in LANDED, reason="P1/P14 not landed")
def test_forbidden_top_level_packages_absent() -> None:
    present = _top_level_packages() & set(FORBIDDEN_TOP_LEVEL_PACKAGES)
    assert not present, (
        f"forbidden top-level packages still present: {sorted(present)}"
    )


@pytest.mark.skipif("closed_layout" not in LANDED, reason="layout not closed")
def test_no_new_top_level_packages_beyond_spec() -> None:
    extra = _top_level_packages() - EXPECTED_TOP_LEVEL_PACKAGES
    assert not extra, (
        f"unexpected top-level packages not in spec 2.1: {sorted(extra)}"
    )


@pytest.mark.parametrize("type_name", SINGLE_DEFINITION_TYPES)
def test_execution_domain_type_defined_once(type_name: str) -> None:
    if type_name == "RunUsage" and "agent_usage_unified" not in LANDED:
        # The agent outcome currently carries its own RunUsage (cost-shaped);
        # the d1b4c91 unification (single domain RunUsage) lands with the
        # agent/engine snapshot rewrite. Hard-gated once that phase completes.
        pytest.xfail("RunUsage duplicate pending agent-usage unification (spec 2.3)")
    rx = re.compile(rf"^class {re.escape(type_name)}\b", re.MULTILINE)
    hits: list[str] = []
    for path in _AI.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if rx.search(text):
            hits.append(str(path.relative_to(_AI)))
    assert len(hits) == 1, (
        f"{type_name} must be defined exactly once, found {len(hits)}: {hits}"
    )


@pytest.mark.parametrize("module", DELETED_MODULES)
def test_deleted_module_is_gone(module: str) -> None:
    if "deletion_manifest" not in LANDED:
        if (_AI / module).exists():
            pytest.xfail(f"{module} pending migration+deletion (spec 2.32)")
    assert not (_AI / module).exists(), (
        f"{module} must be deleted per spec 2.32 deletion manifest"
    )
