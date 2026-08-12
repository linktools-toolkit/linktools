#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local ExecutionBackend backed by AgentExecutor and durable persistence."""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from linktools.core import environ
from pydantic import ValidationError
from pydantic_ai_harness.memory import SearchableMemoryStore
from pydantic_ai_harness.step_persistence import StepStore, continue_run, fork_run

from ..agent import (
    MEMORY_TOOL_NAMES,
    AgentDefinition,
    AgentExecutor,
    select_platform_tool_names,
)
from ..capability import CapabilityRuntimeContext
from ..core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    IdempotencyStatus,
    JsonValue,
    ResourceKind,
    ResourceRef,
    StopReason,
    canonical_json_bytes,
    step_conversation_id,
    step_run_id,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ._execution import CancelEffectOutcome
from ._persistence import (
    ExecutionRecord,
    ExecutionTerminalCommit,
    IdempotencyTerminalUpdate,
    ResultRecord,
    RuntimePersistence,
    SessionHeadAdvance,
)
from ._services import ExecutionRequest

_logger = environ.get_logger("ai.runtime.local")


class LocalExecutionBackend:
    """Resolve immutable definitions and persist one execution lifecycle."""

    def __init__(
        self,
        persistence: RuntimePersistence,
        steps: StepStore,
        executor: AgentExecutor,
        definitions: dict[str, AgentDefinition],
        *,
        tenant_id: str,
        execution_root: Path,
        memory_store_factory: "Callable[[str, str], SearchableMemoryStore] | None" = None,
    ) -> None:
        self._persistence = persistence
        self._steps = steps
        self._executor = executor
        self._definitions = definitions
        self._tenant_id = validate_tenant_id(tenant_id)
        self._execution_root = execution_root
        self._memory_store_factory = memory_store_factory
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._accepting = True

    async def start(self, request: ExecutionRequest, execution: ExecutionRecord) -> None:
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if request.principal.tenant_id != self._tenant_id or execution.tenant_id != self._tenant_id:
            _logger.warning(
                "local execution tenant rejected: expected=%s request=%s execution=%s",
                self._tenant_id,
                request.principal.tenant_id,
                execution.tenant_id,
            )
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if execution.binding_digest not in self._definitions:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        existing = self._tasks.get(execution.execution_id)
        if existing is not None:
            return
        task = asyncio.create_task(self._run(request, execution), name=f"ai-execution-{execution.execution_id}")
        self._tasks[execution.execution_id] = task
        _logger.debug("local execution admitted: execution=%s definition=%s", execution.execution_id, execution.binding_digest)

    async def cancel(self, execution: ExecutionRecord) -> CancelEffectOutcome:
        task = self._tasks.get(execution.execution_id)
        if task is None:
            current = await self._persistence.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
            if current is not None and current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                return CancelEffectOutcome.CONFIRMED
            return CancelEffectOutcome.UNKNOWN
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return CancelEffectOutcome.CONFIRMED

    async def reconcile(self) -> None:
        """Cancel claimed executions whose Harness run was never created."""
        sessions = await self._persistence.sessions.list(tenant_id=self._tenant_id)
        for session in sessions:
            executions = await self._persistence.executions.list_by_session(session.session_id, tenant_id=session.tenant_id)
            for execution in executions:
                if execution.status is not ExecutionStatus.STARTED or execution.agent_run_sequence < 1:
                    continue
                run_id = step_run_id(
                    namespace=self._persistence.namespace,
                    tenant_id=execution.tenant_id,
                    execution_id=execution.execution_id,
                    segment_sequence=execution.agent_run_sequence,
                )
                if await self._steps.get_run(run_id=run_id) is None:
                    _logger.warning(
                        "local recovery cancelling execution without step run: tenant=%s execution=%s",
                        execution.tenant_id,
                        execution.execution_id,
                    )
                    await self._commit_terminal(execution, ExecutionStatus.CANCELLED, None, ErrorCode.EXECUTION_CANCELLED.value, StopReason.CANCELLED)

    async def close(self) -> None:
        self._accepting = False
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _run(self, request: ExecutionRequest, original: ExecutionRecord) -> None:
        execution_id = original.execution_id
        try:
            current = await self._persistence.executions.get(execution_id, tenant_id=original.tenant_id)
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._persistence.executions.claim_next_agent_run(
                execution_id,
                tenant_id=current.tenant_id,
                expected_revision=current.revision,
                expected_agent_run_sequence=current.agent_run_sequence,
            )
            definition = self._definitions.get(current.binding_digest)
            if definition is None:
                raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
            history = await self._history(current)
            run_id = step_run_id(
                namespace=self._persistence.namespace,
                tenant_id=current.tenant_id,
                execution_id=execution_id,
                segment_sequence=current.agent_run_sequence,
            )
            conversation_id = step_conversation_id(
                namespace=self._persistence.namespace,
                tenant_id=current.tenant_id,
                execution_id=execution_id,
            )

            async def sink(event_type: ExecutionEventType, payload: JsonValue) -> None:
                await self._append_event(current, event_type, payload)

            memory = None
            platform_tool_names = select_platform_tool_names(
                allow_tools=definition.spec.allow_tools,
                memory_namespace=current.memory_namespace,
            )
            selected_memory = tuple(name for name in platform_tool_names if name in MEMORY_TOOL_NAMES)
            if selected_memory:
                if self._memory_store_factory is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                memory = self._memory_store_factory(current.tenant_id, current.memory_namespace)
            result = await self._executor.execute(
                definition,
                request.prompt,
                history,
                conversation_id,
                step_store=self._steps,
                step_run_id=run_id,
                segment_sequence=current.agent_run_sequence,
                capability_context=CapabilityRuntimeContext(
                    request.principal,
                    ResourceRef(ResourceKind.EXECUTION, execution_id, current.tenant_id),
                    definition.spec.allow_tools,
                    definition.spec.allow_skills,
                    definition.spec.allow_subagents,
                ),
                memory_namespace=current.memory_namespace,
                memory_store=memory,
                platform_tool_names=platform_tool_names,
                event_sink=sink,
            )
            await self._commit_success(current, definition, result.output, result.run_id)
            _logger.debug("local execution completed: execution=%s run=%s", execution_id, result.run_id)
        except asyncio.CancelledError:
            current = await self._persistence.executions.get(execution_id, tenant_id=original.tenant_id)
            if current is not None and current.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                await self._commit_terminal(current, ExecutionStatus.CANCELLED, None, ErrorCode.EXECUTION_CANCELLED.value, StopReason.CANCELLED)
            raise
        except Exception as error:
            current = await self._persistence.executions.get(execution_id, tenant_id=original.tenant_id)
            if current is not None and current.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                await self._commit_failure(current, error)
            _logger.error("local execution failed: execution=%s", execution_id, exc_info=True)
        finally:
            self._tasks.pop(execution_id, None)

    async def _history(self, execution: ExecutionRecord) -> list[object]:
        if execution.base_execution_id is None:
            return []
        base = await self._persistence.executions.get(execution.base_execution_id, tenant_id=execution.tenant_id)
        if base is None or base.agent_run_sequence < 1:
            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
        run_id = step_run_id(
            namespace=self._persistence.namespace,
            tenant_id=execution.tenant_id,
            execution_id=base.execution_id,
            segment_sequence=base.agent_run_sequence,
        )
        try:
            if execution.lineage_kind is ExecutionLineageKind.FORK:
                return list(await fork_run(self._steps, run_id=run_id))
            return list(await continue_run(self._steps, run_id=run_id))
        except LookupError as error:
            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error

    async def _append_event(self, execution: ExecutionRecord, event_type: ExecutionEventType, payload: JsonValue) -> None:
        current = await self._persistence.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        try:
            await self._persistence.events.append(
                execution.execution_id,
                tenant_id=execution.tenant_id,
                expected_sequence=current.event_sequence,
                event_type=event_type,
                payload=payload,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            latest = await self._persistence.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
            if latest is None:
                raise
            await self._persistence.events.append(
                execution.execution_id,
                tenant_id=execution.tenant_id,
                expected_sequence=latest.event_sequence,
                event_type=event_type,
                payload=payload,
            )

    async def _commit_success(self, execution: ExecutionRecord, definition: AgentDefinition, output: JsonValue, run_id: str) -> None:
        payload = canonical_json_bytes(output)
        blob = await self._persistence.blobs.put_bytes(tenant_id=execution.tenant_id, data=payload)
        current = await self._persistence.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._commit_terminal(
            current,
            ExecutionStatus.SUCCEEDED,
            blob.digest,
            None,
            StopReason.END_TURN,
            definition=definition,
            output_digest=blob.digest,
            run_id=run_id,
        )

    async def _commit_failure(self, execution: ExecutionRecord, error: Exception) -> None:
        code = ErrorCode.OUTPUT_VALIDATION_FAILED if isinstance(error, ValidationError) else error.code if isinstance(error, AIError) else ErrorCode.EXECUTION_FAILED
        await self._commit_terminal(execution, ExecutionStatus.FAILED, None, code.value, StopReason.ERROR)

    async def _commit_terminal(
        self,
        execution: ExecutionRecord,
        status: ExecutionStatus,
        result_ref: str | None,
        error_code: str | None,
        stop_reason: StopReason,
        *,
        definition: AgentDefinition | None = None,
        output_digest: str | None = None,
        run_id: str | None = None,
    ) -> None:
        current = await self._persistence.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
        if current is None or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            return
        now = datetime.now(timezone.utc)
        terminal = _terminal_record(current, status, now, result_ref=result_ref, result_digest=output_digest, error_code=error_code)
        identity = await _terminal_idempotency(self._persistence, current, status, output_digest, error_code)
        if definition is None:
            schema_id, schema_revision, schema_fingerprint = "none", 1, "none"
        else:
            schema_id, schema_revision, schema_fingerprint = definition.spec.output_schema, definition.spec.output_schema_revision, definition.output_schema_fingerprint
        head = None
        if current.session_id is not None and status is ExecutionStatus.SUCCEEDED and current.lineage_kind in {ExecutionLineageKind.SESSION_RESUME, ExecutionLineageKind.RETRY}:
            expected = current.base_execution_id if current.lineage_kind is ExecutionLineageKind.SESSION_RESUME else current.source_execution_id
            head = SessionHeadAdvance(current.session_id, expected, current.execution_id)
        await self._persistence.results.commit_terminal(
            ExecutionTerminalCommit(
                current.revision,
                current.event_sequence,
                terminal,
                ResultRecord(current.execution_id, current.tenant_id, status, schema_id, schema_revision, schema_fingerprint, result_ref, output_digest, stop_reason, 0, 0, 0, now),
                ExecutionEventType.EXECUTION_SUCCEEDED if status is ExecutionStatus.SUCCEEDED else ExecutionEventType.EXECUTION_CANCELLED if status is ExecutionStatus.CANCELLED else ExecutionEventType.EXECUTION_FAILED,
                {"run_id": run_id} if run_id is not None else {"error_code": error_code},
                identity,
                session_head=head,
            )
        )


def _terminal_record(record: ExecutionRecord, status: ExecutionStatus, now: datetime, *, result_ref: str | None, result_digest: str | None, error_code: str | None) -> ExecutionRecord:
    return ExecutionRecord(
        record.execution_id, record.tenant_id, record.session_id, record.binding_digest,
        record.parent_execution_id, record.root_execution_id, record.source_execution_id, record.base_execution_id,
        record.lineage_kind, status, record.revision + 1, record.event_sequence + 1, record.agent_run_sequence,
        result_ref, result_digest, error_code, {}, record.created_at, now, record.memory_namespace,
    )


async def _terminal_idempotency(persistence: RuntimePersistence, execution: ExecutionRecord, status: ExecutionStatus, result_digest: str | None, error_code: str | None) -> IdempotencyTerminalUpdate | None:
    records = await persistence.idempotency.list_by_execution(execution.execution_id, tenant_id=execution.tenant_id)
    if len(records) > 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not records:
        return None
    identity = records[0]
    next_status = IdempotencyStatus.COMPLETED if status is ExecutionStatus.SUCCEEDED else IdempotencyStatus.CANCELLED if status is ExecutionStatus.CANCELLED else IdempotencyStatus.FAILED
    return IdempotencyTerminalUpdate(identity.scope, identity.key_hash, identity.status, next_status, identity.request_digest, result_digest, error_code)


__all__ = ["LocalExecutionBackend"]
