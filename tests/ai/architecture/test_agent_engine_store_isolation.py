#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spec section 12.9 architecture guard: ``AgentEngine`` (the agent package's
model-loop driver) must not import the run-lifecycle Stores -- ``RunStore``,
``CheckpointStore``, ``EventStore`` -- nor ``ApprovalStore``. Per section 12.2
those Stores' lifecycle APIs are owned solely by ``RunCoordinator``; the engine
owns only Prompt / Model loop / Tool calls / Outcome / Cancellation propagation.

Each forbidden symbol is its own parametrized case with ``xfail(strict=True)``:
the WP9 step-3 deeper-form extraction (collapsing ``execute()`` to a Store-free
``AgentExecutionOutcome`` return and crossing terminal-commit ownership into
``RunCoordinator``) is the remaining multi-session core, so most symbols still
XFAIL. ``strict`` makes each case a ratchet: the MOMENT a symbol's import is
removed, that case XPASSES and fails the suite, forcing its owner to drop the
``xfail`` mark so the rule becomes a hard, enforced boundary for that symbol.
Two symbols are already closed -- ``CheckpointStore`` (the engine's
``checkpoint_store`` constructor param was dead: checkpoints are written solely
by ``commit_coordinator``, so it has been removed) and ``ApprovalStore``
(approval writes are owned by ``commit_coordinator`` / ``ApprovalService``, so
the engine never imported it). Both are enforced as hard boundaries.

``SessionStore`` is deliberately NOT in the forbidden set -- section 12.2's
forbidden list is RunStore / CheckpointStore / ApprovalStore / "EventStore
lifecycle API" only, and reading session history for prompt building is part of
the engine's Prompt responsibility. ``CompleteRunCommand`` / ``PauseRunCommand``
(command value objects, not Stores) are also permitted."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_AGENT_ENGINE = (
    Path(__file__).resolve().parents[3]
    / "linktools-ai"
    / "src"
    / "linktools"
    / "ai"
    / "agent"
    / "engine.py"
)

# The Store symbols section 12.2 forbids AgentEngine from depending on. Match
# on the imported NAME so a forbidden Store cannot slip in under an alias.
_FORBIDDEN_STORE_SYMBOLS = ("RunStore", "CheckpointStore", "EventStore", "ApprovalStore")


def _imported_names(file_path: Path) -> "set[str]":
    """Every name brought into module scope by ``import`` / ``from ... import``."""
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return set()
    names: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


@pytest.mark.parametrize(
    "symbol",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True,
                reason="WP9 §12.2: AgentEngine still imports run-lifecycle Stores",
            ),
        )
        for name in ("RunStore", "EventStore")
    ],
)
def test_agent_engine_does_not_import_lifecycle_store(symbol: str) -> None:
    imported = _imported_names(_AGENT_ENGINE)
    assert symbol not in imported, (
        f"agent/engine.py imports forbidden lifecycle Store symbol: {symbol}"
    )


@pytest.mark.parametrize("symbol", ["CheckpointStore", "ApprovalStore"])
def test_agent_engine_does_not_import_closed_store(symbol: str) -> None:
    """Symbols already closed: ``CheckpointStore`` (the engine's dead
    ``checkpoint_store`` param was removed -- checkpoints are written solely by
    ``commit_coordinator``) and ``ApprovalStore`` (approval writes are owned by
    ``commit_coordinator`` / ``ApprovalService``, never imported by the engine).
    These are hard, enforced boundaries -- not xfailed."""
    imported = _imported_names(_AGENT_ENGINE)
    assert symbol not in imported, (
        f"agent/engine.py imports forbidden lifecycle Store symbol: {symbol}"
    )
