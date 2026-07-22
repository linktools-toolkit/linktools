#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""legacy-name inventory -- the baseline that the refactor drives
to zero.

Every pattern below names a concept the refactor eliminates or renames (see
.docs/linktools-ai-architecture-refactor-final-plan-protocol-first-2026-07-18.md
). The test is a *ratchet*: each pattern's file-count must
stay at or below its recorded ceiling. Lower a ceiling as soon as a phase
migrates consumers; the target is 0 across the board by the end of .
Re-introducing a legacy name in new code fails the test, so the inventory
cannot silently regress while the migration is in flight.

Ceilings were captured at the refactor branch point (master bd07cce8 plus the
baseline-repair commit): ``linktools-ai/src`` + ``tests/ai``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SEARCH_ROOTS = (_REPO / "linktools-ai" / "src", _REPO / "tests" / "ai")

# pattern -> ceiling on the number of files that still reference it. Captured
# at the baseline; ratchet downward as phases migrate consumers.
LEGACY_CEILINGS: "dict[str, int]" = {
    "storage.asset": 0,
    # AgentRegistry (+ SkillRegistry/MCPRegistry/ToolRegistry/SwarmRegistry) were
    # the registry-shim classes; the registry/ package is DELETED.
    # All docstring references to the old *Registry names were updated to the
    # current *Catalog names; locked at 0.
    "AgentRegistry": 0,
    # ProviderBundle was renamed to RuntimeDependencies;
    # the old name is fully retired, so the ceiling is locked at 0. The new
    # name (RuntimeDependencies) is NOT tracked here -- it is the current type.
    "ProviderBundle": 0,
    # linktools.ai.providers package DELETED: the
    # spec-provider Protocols moved to their domain homes (agent.spec /
    # swarm.spec / extension.spec / governance.policy.rule / subagent.models /
    # skill.models / mcp.spec) and RuntimeDependencies + ProviderPrefixes +
    # MappingProvider moved to runtime.dependencies (re-exported from
    # linktools.ai.runtime). All src + tests migrated. Locked at 0.
    "linktools.ai.providers": 0,
    # linktools.ai.registry package DELETED: all src + tests
    # migrated to canonical domain homes (agent/skill/mcp/tool/swarm .catalog /
    # .codec / .spec / .models + catalog.parsing). Locked at 0.
    "linktools.ai.registry": 0,
    # linktools.ai.task renamed to linktools.ai.jobs; the package +
    # all consumers + tests are migrated. Locked at 0.
    "linktools.ai.task": 0,
    # linktools.ai.security and linktools.ai.policy merged into
    # linktools.ai.governance.{security,policy}; the packages +
    # all consumers + tests are migrated. Locked at 0. The new names
    # (linktools.ai.governance.security / .policy) are NOT tracked here --
    # they are the current canonical packages.
    "linktools.ai.security": 0,
    "linktools.ai.policy": 0,
    # linktools.ai.knowledge renamed to linktools.ai.retrieval;
    # the package + all consumers + tests are migrated. Locked at 0.
    "linktools.ai.knowledge": 0,
}

# getattr(storage, "assets") is the implicit Artifact fallback 
# explicitly deletes. It must be 0 once lands; today
# there is exactly one occurrence (task/runtime.py).
FORBIDDEN_ZERO: "dict[str, str]" = {
    r'getattr\(storage,\s*["\']assets["\']\)':
        "implicit storage.assets Artifact fallback (§4.8) must be removed",
}


def _matching_files(pattern: str) -> "set[Path]":
    rx = re.compile(pattern)
    matches: "set[Path]" = set()
    for root in _SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            # This file and its sibling guards legitimately mention every
            # pattern as a string literal; scanning them would self-match.
            if "architecture" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if rx.search(text):
                matches.add(path)
    return matches


@pytest.mark.parametrize("pattern, ceiling", list(LEGACY_CEILINGS.items()))
def test_legacy_name_count_does_not_increase(pattern: str, ceiling: int) -> None:
    """Each legacy pattern's file-count must stay at or below its ceiling.

    Lower the ceiling here when a phase migrates consumers; the end state is 0.
    """
    actual = len(_matching_files(rf"\b{re.escape(pattern)}\b"))
    assert actual <= ceiling, (
        f"legacy pattern {pattern!r} grew: {actual} files > ceiling {ceiling}. "
        "Either a new consumer was introduced, or the ceiling should be "
        "lowered to record the migration progress."
    )


@pytest.mark.parametrize("pattern, reason", list(FORBIDDEN_ZERO.items()))
def test_forbidden_patterns_are_absent(pattern: str, reason: str) -> None:
    """Patterns outright forbids must have zero occurrences."""
    files = _matching_files(pattern)
    assert not files, (
        f"{reason}; found in: {[str(p.relative_to(_REPO)) for p in sorted(files)]}"
    )


def test_legacy_inventory_summary(capsys) -> None:
    """Print the current inventory so phase progress is visible under ``-s``.

    This assertion always passes; it exists to surface the counts. As phases
    land, lower the ceilings above until every pattern reads 0.
    """
    lines = ["legacy-name inventory (target: all 0):"]
    for pattern, ceiling in LEGACY_CEILINGS.items():
        actual = len(_matching_files(rf"\b{re.escape(pattern)}\b"))
        lines.append(f"  {pattern:<28} {actual:>3} / ceiling {ceiling}")
    with capsys.disabled():
        print("\n".join(lines))
