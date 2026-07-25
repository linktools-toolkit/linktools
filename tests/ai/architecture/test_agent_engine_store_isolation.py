#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture guard: ``AgentEngine`` (the agent package's Store-free model/tool
loop) must not import ANY run-lifecycle Store -- ``RunStore``, ``SessionStore``,
``EventStore``, ``CheckpointStore``, ``ApprovalStore`` -- nor the
``commit_coordinator`` / ``run_controller``. All Run-lifecycle (RunRecord
create/transition, checkpoint/session/approval persistence, pause/cancel/stream
events, the cross-store commit) is RunCoordinator's sole job; the engine owns
only the prompt-build -> model/tool drive -> outcome path (FS-29).

Each forbidden symbol is its own parametrized case, enforced as a hard
boundary (no xfail). Matching on the imported NAME means a forbidden Store
cannot slip in under an alias. Prior-turn history reaches the engine via
``AgentInput.message_history`` (loaded by RunCoordinator), so ``SessionStore``
is forbidden too -- the engine never reads session persistence itself."""

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

# The Store symbols AgentEngine must not depend on. Match on the imported NAME
# so a forbidden Store cannot slip in under an alias.
_FORBIDDEN_STORE_SYMBOLS = (
    "RunStore",
    "SessionStore",
    "EventStore",
    "CheckpointStore",
    "ApprovalStore",
)


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


@pytest.mark.parametrize("symbol", _FORBIDDEN_STORE_SYMBOLS)
def test_agent_engine_does_not_import_lifecycle_store(symbol: str) -> None:
    """FS-29: AgentEngine must not import any run-lifecycle Store. Hard
    boundary -- a regression that re-imports one fails this test outright."""
    imported = _imported_names(_AGENT_ENGINE)
    assert symbol not in imported, (
        f"agent/engine.py imports forbidden lifecycle Store symbol: {symbol}"
    )


def test_agent_engine_signature_rejects_lifecycle_params() -> None:
    """FS-29: AgentEngine.__init__ must not accept any run-lifecycle Store,
    commit_coordinator, or run_controller parameter. Guards against a regression
    that re-adds one of the removed constructor knobs (the engine's only inputs
    are pure-execution deps: middleware/memory/retriever/sandbox/capability/
    security/metrics/pricing)."""
    import inspect

    from linktools.ai.agent.engine import AgentEngine

    params = inspect.signature(AgentEngine.__init__).parameters
    forbidden = {
        "run_store",
        "session_store",
        "event_store",
        "checkpoint_store",
        "approval_store",
        "commit_coordinator",
        "run_controller",
    }
    present = forbidden & set(params)
    assert not present, (
        f"AgentEngine.__init__ must not accept run-lifecycle params, "
        f"got {sorted(present)}"
    )
