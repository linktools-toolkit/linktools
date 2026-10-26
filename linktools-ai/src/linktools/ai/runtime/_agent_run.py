#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local AgentRun execution primitives."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..core import JsonValue
from ..errors import ErrorCode, ErrorDiagnostics
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.runs import CheckpointPersistence
from linktools.ai.execution import ExecutionRecord
from linktools.ai.persistence.runs import ExecutionTerminalSealPlan


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    code: ErrorCode
    safe_details: Mapping[str, JsonValue]
    diagnostics: ErrorDiagnostics | None = None
    category: str | None = None
    retryable: bool | None = None
    operation_id: str | None = None


class _AgentRunLifecycle(Protocol):
    def execution_producer_generation(
        self, agent_run_id: str, *, execution_id: str,
    ) -> int | None: ...
    async def materialize_conversation(self, *, agent_run_id: str) -> None: ...
    async def materialize_from_recovery(
        self,
        *,
        target: PersistenceDomain,
        agent_run_id: str,
        execution_id: "str | None" = None,
    ) -> None: ...
    @property
    def checkpoint_persistence(self) -> CheckpointPersistence: ...
    async def verify_terminal_attempts(
        self,
        *,
        candidate_agent_run_ids: tuple[str, ...],
        required_agent_run_id: str | None,
    ) -> None: ...
    async def release_staging_many(
        self,
        *,
        candidate_agent_run_ids: tuple[str, ...],
        execution_id: "str | None" = None,
    ) -> None: ...
    async def flush_execution_projection(
        self, agent_run_id: str, *, execution_id: str, producer_generation: int | None = None,
    ) -> None: ...
    async def discard_revoked_producer_staging(
        self, agent_run_id: str, *, execution_id: str, producer_generation: int,
        terminal_execution: ExecutionRecord | None = None,
    ) -> None: ...
    async def wait_projection_flight(self, agent_run_id: str) -> None: ...
    async def prepare_execution_terminal_seal(
        self,
        *,
        execution_id: str,
        agent_run_ids: Sequence[str],
        binding_digest: str,
    ) -> ExecutionTerminalSealPlan: ...
    async def finalize_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None: ...
    async def reconcile_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None: ...
    async def discard_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None: ...
