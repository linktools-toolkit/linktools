#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Commit preferences are immutable and restricted by each persistence owner."""

from dataclasses import FrozenInstanceError

import pytest

from linktools.ai.capability import ConversationCapability, ExecutionCapability, TraceCapability
from linktools.ai.core import DEFAULT_TRACE_BATCH_SIZE, DEFAULT_TRACE_FLUSH_INTERVAL
from linktools.ai.core import CommitPolicy


def test_commit_policy_defaults_and_immutable_shared_declarations() -> None:
    assert ExecutionCapability().commit == CommitPolicy.sync()
    assert TraceCapability().commit == CommitPolicy.batch(
        batch_size=DEFAULT_TRACE_BATCH_SIZE,
        flush_interval=DEFAULT_TRACE_FLUSH_INTERVAL,
    )
    policy = CommitPolicy.terminal()
    first = ExecutionCapability(commit=policy)
    second = ExecutionCapability(commit=policy)
    assert first.commit is second.commit is policy
    with pytest.raises(FrozenInstanceError):
        policy.mode = "sync"


@pytest.mark.parametrize("settings", [
    {"mode": "unknown"},
    {"mode": "sync", "batch_size": 1},
    {"mode": "terminal", "flush_interval": 1},
    {"mode": "batch"},
    {"mode": "batch", "batch_size": True, "flush_interval": 1},
    {"mode": "batch", "batch_size": 1, "flush_interval": float("nan")},
    {"mode": "batch", "batch_size": 1, "flush_interval": float("inf")},
])
def test_commit_policy_rejects_invalid_declarations(settings: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CommitPolicy(**settings)


def test_persistence_owners_reject_unsupported_policy_modes_and_removed_arguments() -> None:
    with pytest.raises(ValueError):
        ExecutionCapability(commit=CommitPolicy.batch(batch_size=2, flush_interval=1))
    with pytest.raises(ValueError):
        TraceCapability(commit=CommitPolicy.terminal())
    with pytest.raises(TypeError):
        ConversationCapability(commit=CommitPolicy.sync())
    with pytest.raises(TypeError):
        ConversationCapability(cache=None)
    with pytest.raises(TypeError):
        ConversationCapability(storage=None)
    with pytest.raises(TypeError):
        TraceCapability(batch_size=2)
    with pytest.raises(TypeError):
        TraceCapability(flush_interval=1)


def test_execution_commit_has_one_public_configuration_owner() -> None:
    import inspect
    from linktools.ai.runtime import Runtime
    assert "commit" not in inspect.signature(Runtime.open).parameters
    assert "commit" in inspect.signature(ExecutionCapability).parameters
