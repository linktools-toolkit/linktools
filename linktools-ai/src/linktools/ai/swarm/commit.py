#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swarm commit boundary (P7 swarm commit boundary).

The SwarmCommitCoordinator owns the cross-store commit for every swarm +
swarm-step lifecycle point: state transition, state-critical event append,
and (for terminal commits) the commit-log entry that makes the operation
idempotent by commit_id. SwarmEngine receives the coordinator (along with
AgentCompiler, RunDispatcher, Clock, SwarmEventSink) and never touches a
RunStore / SessionStore / EventStore / RunDefinitionStore / append_event /
mark_completed / mark_failed directly -- the dep-cleanup rule.

The Protocol is the swarm-side mirror of RunCommitCoordinator. Each command
carries its deterministic commit_id, the swarm_run_id it targets, the
expected version for optimistic concurrency, an execution fence or owner
token (when the topology requires fencing), the typed payload, and the
lifecycle event context. Same commit_id + same payload = idempotent replay;
same commit_id + different payload = SwarmCommitConflictError.

The reference SQL implementation groups swarm-state, step-state, event, and
commit-log writes inside one Storage UoW so a swarm commit is all-or-nothing.
The Filesystem implementation journals the same shape."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class StartSwarmCommand:
    commit_id: str
    swarm_run_id: str
    expected_version: int
    payload: Mapping[str, Any]
    event_context: Any


@dataclass(frozen=True, slots=True)
class StartSwarmStepCommand:
    commit_id: str
    swarm_run_id: str
    step_attempt_id: str
    expected_version: int
    payload: Mapping[str, Any]
    event_context: Any


@dataclass(frozen=True, slots=True)
class CompleteSwarmStepCommand:
    commit_id: str
    swarm_run_id: str
    step_attempt_id: str
    expected_version: int
    payload: Mapping[str, Any]
    event_context: Any


@dataclass(frozen=True, slots=True)
class FailSwarmStepCommand:
    commit_id: str
    swarm_run_id: str
    step_attempt_id: str
    expected_version: int
    payload: Mapping[str, Any]
    event_context: Any


@dataclass(frozen=True, slots=True)
class CompleteSwarmCommand:
    commit_id: str
    swarm_run_id: str
    expected_version: int
    payload: Mapping[str, Any]
    event_context: Any


@dataclass(frozen=True, slots=True)
class FailSwarmCommand:
    commit_id: str
    swarm_run_id: str
    expected_version: int
    payload: Mapping[str, Any]
    event_context: Any


@dataclass(frozen=True, slots=True)
class CancelSwarmCommand:
    commit_id: str
    swarm_run_id: str
    expected_version: int
    payload: Mapping[str, Any]
    event_context: Any


@runtime_checkable
class SwarmCommitCoordinator(Protocol):
    """Atomic cross-store commit for swarm + swarm-step lifecycle points.
    Each method runs the state transition + state-critical event append +
    commit-log entry in ONE Storage UoW so the commit is all-or-nothing,
    and is idempotent by commit_id (same id + same payload returns the
    recorded result; same id + different payload is a conflict)."""

    async def start(self, command: StartSwarmCommand) -> Any: ...
    async def start_step(self, command: StartSwarmStepCommand) -> Any: ...
    async def complete_step(self, command: CompleteSwarmStepCommand) -> Any: ...
    async def fail_step(self, command: FailSwarmStepCommand) -> Any: ...
    async def complete(self, command: CompleteSwarmCommand) -> Any: ...
    async def fail(self, command: FailSwarmCommand) -> Any: ...
    async def cancel(self, command: CancelSwarmCommand) -> Any: ...


class SwarmCommitConflictError(Exception):
    """A retried swarm commit used the SAME commit_id with a DIFFERENT payload.
    Same-id-same-hash is an idempotent replay (returns the recorded result);
    same-id-different-hash is this conflict -- the caller is asserting two
    distinct operations under one id, which the commit log refuses to
    silently collapse."""


__all__: "list[str]" = [
    "CancelSwarmCommand",
    "CompleteSwarmCommand",
    "CompleteSwarmStepCommand",
    "FailSwarmCommand",
    "FailSwarmStepCommand",
    "StartSwarmCommand",
    "StartSwarmStepCommand",
    "SwarmCommitConflictError",
    "SwarmCommitCoordinator",
]
