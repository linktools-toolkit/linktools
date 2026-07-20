#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Holistic-closure contracts: the invariants each remaining WP must establish.

Each contract is written to FAIL today and marked ``xfail(strict=True)``; when
its WP lands the test begins to pass, strict-xfail turns it red, and the marker
must be removed -- so the fence cannot be silently dropped. Do NOT delete a
failing test to make the suite green; remove the marker only when the gain is
real."""

import inspect
from pathlib import Path

import pytest

_AI_SRC = (
    Path(__file__).resolve().parents[3] / "linktools-ai" / "src" / "linktools" / "ai"
)


def _src_text() -> str:
    parts = []
    for p in _AI_SRC.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts)


# --- WP-01: checkpoint sequencing is Store-owned ---------------------------


def test_checkpoint_callers_do_not_hardcode_sequence():
    """Checkpoint construction is owned by the RunCommitCoordinators (Store
    assigns the sequence); the AgentEngine delegates and never constructs a
    checkpoint itself, and no caller builds the persisted RunCheckpoint with a
    caller-owned sequence. The Store implementations legitimately assign the
    first sequence; this contract targets callers."""
    import re

    runner = (_AI_SRC / "agent" / "runner.py").read_text(encoding="utf-8")
    assert "NewRunCheckpoint(" not in runner, (
        "AgentEngine must not construct checkpoints directly -- the "
        "RunCommitCoordinator owns checkpoint creation"
    )
    commit_files = [
        (_AI_SRC / "storage" / "filesystem" / "commit.py"),
        (_AI_SRC / "storage" / "sqlalchemy" / "commit.py"),
    ]
    for commit_file in commit_files:
        text = commit_file.read_text(encoding="utf-8")
        assert "NewRunCheckpoint(" in text, (
            f"{commit_file.name} must construct NewRunCheckpoint (Store-owned sequence)"
        )
        persisted = [
            i
            for i in (m.start() for m in re.finditer(r"\bRunCheckpoint\s*\(", text))
            if text[max(0, i - 3) : i] != "New"
        ]
        assert not persisted, (
            f"{commit_file.name} constructs persisted RunCheckpoint (caller-owned sequence)"
        )


# --- WP-03: single approval path -------------------------------------------


def test_runtime_build_has_no_pause_on_approval():
    from linktools.ai.runtime import Runtime

    build_params = inspect.signature(Runtime.build).parameters
    assert "pause_on_approval" not in build_params, (
        "Runtime.build must not expose pause_on_approval (single pause path)"
    )


def test_agent_runner_requires_commit_coordinator():
    """AgentEngine.commit_coordinator has no default -- the cross-store commit
    is coordinator-owned and Runtime.build always wires one. There is no inline
    fallback path."""
    from linktools.ai.agent.runner import AgentEngine

    param = inspect.signature(AgentEngine.__init__).parameters["commit_coordinator"]
    assert param.default is inspect.Parameter.empty, (
        "AgentEngine.commit_coordinator must be required (no inline commit fallback)"
    )


def test_agent_runner_has_no_inline_commit_params():
    """The runner no longer accepts the old inline-commit knobs (uow_factory,
    approval_store) -- a single RunCommitCoordinator owns pause/complete."""
    from linktools.ai.agent.runner import AgentEngine

    params = inspect.signature(AgentEngine.__init__).parameters
    forbidden = {"uow_factory", "approval_store"}
    assert not (forbidden & set(params)), (
        f"AgentEngine must not accept inline-commit params, "
        f"got {sorted(forbidden & set(params))}"
    )


# --- WP-04: resume does not accept a caller spec / identity ----------------


def test_runtime_resume_takes_only_run_id():
    from linktools.ai.runtime import Runtime

    params = inspect.signature(Runtime.resume).parameters
    forbidden = {"spec", "user_id", "tenant_id", "workspace"}
    assert not (forbidden & set(params)), (
        f"Runtime.resume must not accept caller-supplied identity/spec, "
        f"got {sorted(forbidden & set(params))}"
    )


# --- WP-05: swarm resume does not accept a caller spec ---------------------


def test_swarm_resume_takes_only_swarm_run_id():
    from linktools.ai.swarm.runner import SwarmRunner

    params = inspect.signature(SwarmRunner.resume).parameters
    forbidden = {"spec", "agents", "user_id", "tenant_id"}
    assert not (forbidden & set(params)), (
        f"SwarmRunner.resume must not accept caller spec/agents/identity, "
        f"got {sorted(forbidden & set(params))}"
    )


# --- WP-07: idempotency is claim/owner/generation --------------------------


def test_idempotency_store_exposes_claim():
    from linktools.ai.tool.idempotency import IdempotencyStore

    assert hasattr(IdempotencyStore, "claim"), (
        "IdempotencyStore must expose claim (owner/generation/lease model)"
    )


# --- WP-09: domain models validate at construction -------------------------


def test_core_domain_models_enforce_invariants():
    """Core domain models enforce their contract at construction (not just via
    the registry parser), so a custom provider building one directly cannot
    create an invalid object. Covers the Agent + Tool + ModelPolicy domains."""
    from linktools.ai.agent.spec import ToolRef
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.tool.models import ToolDescriptor

    with pytest.raises(ValueError):
        ModelPolicy(primary="")
    with pytest.raises(ValueError):
        ToolRef(kind="  ", name="n")
    with pytest.raises(ValueError):
        ToolDescriptor(name="", source="s", category="c", risk="low", mutating=False)
    with pytest.raises(TypeError):
        ToolDescriptor(name="t", source="s", category="c", risk="low", mutating="yes")


# --- WP-08: no default=str canonicalization --------------------------------


def test_no_default_str_in_canonical_paths():
    """The ``default`` stringification argument to ``json.dumps`` is forbidden:
    it makes canonical JSON unstable (silently coerces arbitrary objects). All
    canonical/hash/fingerprint paths use linktools.ai.json.canonical_json."""
    text = _src_text()
    assert "default=str" not in text, "default=str still present in src"


# --- WP-13: model security pipeline is wired -------------------------------


def test_runner_invokes_model_security_hooks():
    """The security pipeline fires per model REQUEST via a SecuredModel wrapper
    the runner passes to Agent.iter(model=...) -- not just once around the run.
    The runner wires the wrapper; the wrapper holds the before_model/after_model
    calls."""
    from linktools.ai.agent.runner import AgentEngine
    from linktools.ai.governance.security.secured_model import SecuredModel

    runner_src = inspect.getsource(AgentEngine)
    assert "SecuredModel" in runner_src, (
        "AgentEngine must wrap the model with SecuredModel for per-request hooks"
    )
    wrapper_src = inspect.getsource(SecuredModel)
    assert "before_model(" in wrapper_src and "after_model(" in wrapper_src, (
        "SecuredModel must invoke before_model/after_model around each request"
    )


# --- WP-14: budget is not deferred -----------------------------------------


def test_budget_is_enforced():
    text = _src_text()
    assert "deferred" not in text.lower(), (
        "budget/max_total_cost still described as deferred; must be enforced"
    )


# --- WP-16: streaming does not swallow exceptions --------------------------


def test_streaming_does_not_swallow_exceptions():
    """The streaming blocks must not swallow exceptions with a bare
    ``except Exception: ... pass``. Real stream errors propagate; the only
    caught case is the specific non-streaming-model signal (AssertionError)."""
    import re

    src = (
        Path(__file__).resolve().parents[3]
        / "linktools-ai"
        / "src"
        / "linktools"
        / "ai"
        / "agent"
        / "runner.py"
    ).read_text(encoding="utf-8")
    swallow = re.findall(r"except Exception:\s*\n(?:\s*#[^\n]*\n)*\s*pass\b", src)
    assert not swallow, (
        f"AgentEngine still swallows exceptions via 'except Exception: pass' "
        f"({len(swallow)} site(s))"
    )
