#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunCommitCoordinator: the protocol for atomically committing the cross-store
state transitions a Run's pause or completion requires.

The SQL backend implements this via a shared UnitOfWork; the File backend via
sequential writes + fail-closed propagation."""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class PauseRunCommand:
    run_id: str
    expected_version: int
    approval_request: Mapping[str, Any]
    checkpoint_payload: bytes
    event_context: Any
    # Deterministic id (caller sets e.g. ``pause:{run_id}:{approval_id}``) so a
    # retried pause is idempotent -- the coordinator recognizes an already-
    # committed pause and writes nothing instead of duplicating artifacts.
    commit_id: str = ""


@dataclass(frozen=True, slots=True)
class CompleteRunCommand:
    run_id: str
    session_id: str
    expected_version: int
    messages: "tuple[Any, ...]"
    checkpoint_payload: bytes
    result: Any
    event_context: Any
    # Deterministic id (caller sets e.g. ``complete:{run_id}:{expected_version}``)
    # so a retried complete is idempotent.
    commit_id: str = ""


@dataclass(frozen=True, slots=True)
class PausedRunCommit:
    approval_id: str
    checkpoint_id: str


@dataclass(frozen=True, slots=True)
class CompletedRunCommit:
    result: Any


@runtime_checkable
class RunCommitCoordinator(Protocol):
    async def pause(self, command: PauseRunCommand) -> PausedRunCommit: ...

    async def complete(self, command: CompleteRunCommand) -> CompletedRunCommit: ...
