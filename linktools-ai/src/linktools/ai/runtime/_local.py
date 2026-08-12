#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local ExecutionBackend backed by AgentExecutor and durable persistence."""

import asyncio
from collections.abc import Callable
from dataclasses import replace
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
    Principal,
    StopReason,
    canonical_json_bytes,
    step_conversation_id,
    step_run_id,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..storage import StorageDomain
from ._execution import CancelEffectOutcome
from ._persistence import (
    ConversationCursor,
    ExecutionRecord,
    ExecutionTerminalCommit,
    IdempotencyTerminalUpdate,
    ResultRecord,
    RecoveryCheckpoint,
    RuntimeStores,
)
from ._services import ExecutionRequest

_logger = environ.get_logger("ai.runtime.local")


class LocalExecutionBackend:
    """Resolve immutable definitions and persist one execution lifecycle."""

    def __init__(
        self,
        persistence: RuntimeStores,
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
            current = await self._persistence.execution.get(execution.execution_id, tenant_id=execution.tenant_id)
            if current is not None and current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                return CancelEffectOutcome.CONFIRMED
            return CancelEffectOutcome.UNKNOWN
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return CancelEffectOutcome.CONFIRMED

    async def reconcile(self) -> None:
        """Rebuild transient execution state from recovery-owned checkpoints."""
        checkpoints = await self._persistence.recovery_checkpoint.list(tenant_id=self._tenant_id)
        for checkpoint in checkpoints:
            if checkpoint.phase in {"completed", "failed", "cancelled"}:
                continue
            payload = checkpoint.payload
            prompt = payload.get("prompt")
            principal_id = payload.get("principal_id")
            principal_kind = payload.get("principal_kind")
            if not isinstance(prompt, str) or not isinstance(principal_id, str) or not isinstance(principal_kind, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            principal = Principal(principal_id, checkpoint.tenant_id, principal_kind)
            execution = await self._persistence.execution.get(checkpoint.execution_id, tenant_id=checkpoint.tenant_id)
            if execution is not None and execution.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                phase = "completed" if execution.status is ExecutionStatus.SUCCEEDED else execution.status.value.lower()
                await self._finish_checkpoint(checkpoint, phase)
                continue
            if execution is None:
                sequence = payload.get("agent_run_sequence")
                if not isinstance(sequence, int) or sequence < 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                session_id = payload.get("session_id")
                if isinstance(session_id, str):
                    session = await self._persistence.conversation.get(session_id, tenant_id=checkpoint.tenant_id)
                    if session is None:
                        session_id = None
                execution = ExecutionRecord(
                    checkpoint.execution_id,
                    checkpoint.tenant_id,
                    session_id if isinstance(session_id, str) else None,
                    checkpoint.binding_digest,
                    None,
                    checkpoint.execution_id,
                    None,
                    None,
                    ExecutionLineageKind.RUN,
                    ExecutionStatus.STARTED,
                    0,
                    0,
                    sequence,
                    None,
                    None,
                    None,
                    {},
                    checkpoint.created_at,
                    checkpoint.updated_at,
                    payload.get("memory_namespace") if isinstance(payload.get("memory_namespace"), str) else None,
                    checkpoint.run_id,
                )
                await self._persistence.execution.create(execution)
            request = ExecutionRequest(
                prompt=prompt,
                principal=principal,
                idempotency_key=f"recovery:{checkpoint.checkpoint_id}",
                memory_namespace=payload.get("memory_namespace") if isinstance(payload.get("memory_namespace"), str) else None,
            )
            await self.start(request, execution)
            _logger.info("local recovery execution relaunched: tenant=%s checkpoint=%s execution=%s", checkpoint.tenant_id, checkpoint.checkpoint_id, checkpoint.execution_id)

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
        checkpoint: RecoveryCheckpoint | None = None
        try:
            current = await self._persistence.execution.get(execution_id, tenant_id=original.tenant_id)
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._persistence.execution.claim_next_agent_run(
                execution_id,
                tenant_id=current.tenant_id,
                expected_revision=current.revision,
                expected_agent_run_sequence=current.agent_run_sequence,
            )
            definition = self._definitions.get(current.binding_digest)
            if definition is None:
                raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
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
            now = datetime.now(timezone.utc)
            existing_checkpoint = await self._persistence.recovery_checkpoint.get(execution_id, tenant_id=current.tenant_id)
            if existing_checkpoint is not None and existing_checkpoint.phase not in {"completed", "failed", "cancelled"}:
                checkpoint = existing_checkpoint
                history_run_id = checkpoint.payload.get("history_run_id")
                if not isinstance(history_run_id, str):
                    history_run_id = checkpoint.run_id
                snapshot = await self._steps.latest_snapshot(run_id=history_run_id, include_interrupted=True)
                current = replace(current, conversation_run_id=history_run_id if snapshot is not None else None)
                if current.session_id is not None:
                    session = await self._persistence.conversation.get(current.session_id, tenant_id=current.tenant_id)
                    if session is None:
                        current = replace(current, session_id=None)
            else:
                checkpoint = await self._persistence.recovery_checkpoint.create(
                    RecoveryCheckpoint(
                        execution_id,
                        execution_id,
                        current.tenant_id,
                        current.binding_digest,
                        run_id,
                        "running",
                        f"execution:{execution_id}",
                        {
                            "agent_run_sequence": current.agent_run_sequence,
                            "prompt": request.prompt,
                            "principal_id": request.principal.principal_id,
                            "principal_kind": request.principal.kind,
                            "session_id": current.session_id,
                            "memory_namespace": current.memory_namespace,
                            "agent_id": definition.spec.id,
                            "prompt_id": definition.prompt.id,
                            "history_run_id": run_id,
                            "idempotency": {"key": request.idempotency_key, "status": "running"},
                            "approval": {"pending": []},
                            "external": {"pending": []},
                            "tool_effect": {"pending": []},
                            "terminal_handoff": None,
                        },
                        0,
                        now,
                        now,
                        {"pending": []},
                        {"pending": []},
                        {"pending": []},
                        {"key": request.idempotency_key, "status": "running"},
                        None,
                    )
                )
            history = await self._history(current)

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
            await self._finish_checkpoint(checkpoint, "completed")
            _logger.debug("local execution completed: execution=%s run=%s", execution_id, result.run_id)
        except asyncio.CancelledError:
            current = await self._persistence.execution.get(execution_id, tenant_id=original.tenant_id)
            if current is not None and current.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                await self._commit_terminal(current, ExecutionStatus.CANCELLED, None, ErrorCode.EXECUTION_CANCELLED.value, StopReason.CANCELLED)
            if checkpoint is not None:
                await self._finish_checkpoint(checkpoint, "cancelled")
            raise
        except Exception as error:
            current = await self._persistence.execution.get(execution_id, tenant_id=original.tenant_id)
            if current is not None and current.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                await self._commit_failure(current, error)
            if checkpoint is not None:
                await self._finish_checkpoint(checkpoint, "failed")
            _logger.error("local execution failed: execution=%s", execution_id, exc_info=True)
        finally:
            self._tasks.pop(execution_id, None)

    async def _finish_checkpoint(self, checkpoint: RecoveryCheckpoint, phase: str) -> None:
        if checkpoint.phase == phase:
            return
        payload = dict(checkpoint.payload)
        handoff = {"phase": phase, "execution_id": checkpoint.execution_id}
        payload["terminal_handoff"] = handoff
        idempotency = payload.get("idempotency")
        if isinstance(idempotency, dict):
            idempotency = dict(idempotency)
            idempotency["status"] = phase
            payload["idempotency"] = idempotency
        updated = replace(
            checkpoint,
            phase=phase,
            payload=payload,
            idempotency_state=idempotency if isinstance(idempotency, dict) else checkpoint.idempotency_state,
            terminal_handoff=handoff,
            revision=checkpoint.revision + 1,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            await self._persistence.recovery_checkpoint.compare_and_swap(
                checkpoint.checkpoint_id,
                tenant_id=checkpoint.tenant_id,
                expected_revision=checkpoint.revision,
                next_record=updated,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.recovery_checkpoint.get(checkpoint.checkpoint_id, tenant_id=checkpoint.tenant_id)
            if current is None or current.phase != phase:
                raise

    async def _history(self, execution: ExecutionRecord) -> list[object]:
        if execution.conversation_run_id is not None:
            try:
                return list(await continue_run(self._steps, run_id=execution.conversation_run_id, include_interrupted=True))
            except LookupError as error:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error
        if execution.base_execution_id is None:
            return []
        base = await self._persistence.execution.get(execution.base_execution_id, tenant_id=execution.tenant_id)
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
        current = await self._persistence.execution.get(execution.execution_id, tenant_id=execution.tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        try:
            await self._persistence.execution_events.append(
                execution.execution_id,
                tenant_id=execution.tenant_id,
                expected_sequence=current.event_sequence,
                event_type=event_type,
                payload=payload,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            latest = await self._persistence.execution.get(execution.execution_id, tenant_id=execution.tenant_id)
            if latest is None:
                raise
            await self._persistence.execution_events.append(
                execution.execution_id,
                tenant_id=execution.tenant_id,
                expected_sequence=latest.event_sequence,
                event_type=event_type,
                payload=payload,
            )

    async def _commit_success(self, execution: ExecutionRecord, definition: AgentDefinition, output: JsonValue, run_id: str) -> None:
        payload = canonical_json_bytes(output)
        blob = await self._persistence.blobs.for_domain(StorageDomain.EXECUTION).put_bytes(tenant_id=execution.tenant_id, data=payload)
        current = await self._persistence.execution.get(execution.execution_id, tenant_id=execution.tenant_id)
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
        current = await self._persistence.execution.get(execution.execution_id, tenant_id=execution.tenant_id)
        if current is None or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            return
        now = datetime.now(timezone.utc)
        terminal = _terminal_record(current, status, now, result_ref=result_ref, result_digest=output_digest, error_code=error_code)
        identity = await _terminal_idempotency(self._persistence, current, status, output_digest, error_code)
        if definition is None:
            schema_id, schema_revision, schema_fingerprint = "none", 1, "none"
        else:
            schema_id, schema_revision, schema_fingerprint = definition.spec.output_schema, definition.spec.output_schema_revision, definition.output_schema_fingerprint
        if current.session_id is not None and status is ExecutionStatus.SUCCEEDED and run_id is not None:
            session = await self._persistence.conversation.get(current.session_id, tenant_id=current.tenant_id)
            if session is None:
                checkpoint = await self._persistence.recovery_checkpoint.get(current.execution_id, tenant_id=current.tenant_id)
                if checkpoint is None or checkpoint.phase in {"completed", "failed", "cancelled"}:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                _logger.info(
                    "recovery terminal handoff without conversation cursor: execution=%s checkpoint=%s",
                    current.execution_id,
                    checkpoint.checkpoint_id,
                )
                session = None
            if session is not None:
                next_cursor = ConversationCursor(run_id, terminal.revision)
                if session.continuation != next_cursor:
                    try:
                        await self._persistence.conversation.advance_continuation(
                            current.session_id,
                            tenant_id=current.tenant_id,
                            expected=session.continuation,
                            next_cursor=next_cursor,
                        )
                    except AIError as error:
                        if error.code is not ErrorCode.STORAGE_CONFLICT:
                            raise
                        latest = await self._persistence.conversation.get(current.session_id, tenant_id=current.tenant_id)
                        if latest is None or latest.continuation != next_cursor:
                            raise
        await self._persistence.execution_result.commit_terminal(
            ExecutionTerminalCommit(
                current.revision,
                current.event_sequence,
                terminal,
                ResultRecord(current.execution_id, current.tenant_id, status, schema_id, schema_revision, schema_fingerprint, result_ref, output_digest, stop_reason, 0, 0, 0, now),
                ExecutionEventType.EXECUTION_SUCCEEDED if status is ExecutionStatus.SUCCEEDED else ExecutionEventType.EXECUTION_CANCELLED if status is ExecutionStatus.CANCELLED else ExecutionEventType.EXECUTION_FAILED,
                {"run_id": run_id} if run_id is not None else {"error_code": error_code},
                identity,
            )
        )


def _terminal_record(record: ExecutionRecord, status: ExecutionStatus, now: datetime, *, result_ref: str | None, result_digest: str | None, error_code: str | None) -> ExecutionRecord:
    return ExecutionRecord(
        record.execution_id, record.tenant_id, record.session_id, record.binding_digest,
        record.parent_execution_id, record.root_execution_id, record.source_execution_id, record.base_execution_id,
        record.lineage_kind, status, record.revision + 1, record.event_sequence + 1, record.agent_run_sequence,
        result_ref, result_digest, error_code, {}, record.created_at, now, record.memory_namespace, record.conversation_run_id,
    )


async def _terminal_idempotency(persistence: RuntimeStores, execution: ExecutionRecord, status: ExecutionStatus, result_digest: str | None, error_code: str | None) -> IdempotencyTerminalUpdate | None:
    records = await persistence.execution_idempotency.list_by_execution(execution.execution_id, tenant_id=execution.tenant_id)
    if len(records) > 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not records:
        return None
    identity = records[0]
    next_status = IdempotencyStatus.COMPLETED if status is ExecutionStatus.SUCCEEDED else IdempotencyStatus.CANCELLED if status is ExecutionStatus.CANCELLED else IdempotencyStatus.FAILED
    return IdempotencyTerminalUpdate(identity.scope, identity.key_hash, identity.status, next_status, identity.request_digest, result_digest, error_code)


__all__ = ["LocalExecutionBackend"]
