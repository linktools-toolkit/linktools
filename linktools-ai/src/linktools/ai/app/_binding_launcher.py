#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process-local launcher for immutable Agent bindings."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.core import environ
from pydantic import ValidationError

from ..adapter import RuntimeMemoryStore
from ..agent import (
    AgentBinding,
    AgentBindingRegistry,
    BoundAgentRunner,
    EventHandler,
    WorkspaceAgentResult,
)
from ..capability import CapabilityRuntimeContext
from ..core import (
    ExecutionEventType,
    ExecutionStatus,
    IdempotencyStatus,
    JsonValue,
    ResourceKind,
    ResourceRef,
    StopReason,
    step_conversation_id,
    step_run_id,
)
from ..errors import AIError, ErrorCode
from ..model import ModelMaterializer
from ..runtime import (
    CancelEffectOutcome,
    ExecutionRecord,
    ExecutionRequest,
    ExecutionTerminalCommit,
    IdempotencyTerminalUpdate,
    ResultRecord,
)

if TYPE_CHECKING:
    from ._assembly import RuntimeResources


_logger = environ.get_logger("ai.app.binding")


class BindingExecutionLauncher:
    """Run registered immutable Agent bindings against shared Runtime persistence."""

    def __init__(
        self,
        registry: AgentBindingRegistry,
        materializer: ModelMaterializer,
        resources: "RuntimeResources",
        *,
        execution_root: "Path",
        event_handler: "EventHandler | None" = None,
    ) -> None:
        self._registry = registry
        self._materializer = materializer
        self._resources = resources
        self._execution_root = execution_root
        self._event_handler = event_handler
        self._tasks: dict[str, asyncio.Task[WorkspaceAgentResult]] = {}
        self._accepting = True

    async def start(self, request: ExecutionRequest, execution: ExecutionRecord) -> None:
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        binding = self._registry.resolve(execution.binding_digest)
        if execution.execution_id in self._tasks:
            return
        self._tasks[execution.execution_id] = asyncio.create_task(self._execute(binding, request, execution))
        _logger.debug("binding launch registered: execution=%s binding=%s", execution.execution_id, execution.binding_digest)

    def validate_binding(self, binding_digest: str) -> None:
        self._registry.resolve(binding_digest)

    async def cancel(self, execution: ExecutionRecord) -> CancelEffectOutcome:
        task = self._tasks.get(execution.execution_id)
        if task is None:
            return CancelEffectOutcome.UNKNOWN
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        _logger.debug("binding cancellation resolved: execution=%s", execution.execution_id)
        return CancelEffectOutcome.CONFIRMED

    async def shutdown(self) -> None:
        self._accepting = False
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _execute(self, binding: AgentBinding, request: ExecutionRequest, execution: ExecutionRecord) -> WorkspaceAgentResult:
        runner = BoundAgentRunner(
            binding=binding,
            materializer=self._materializer,
            execution_root=self._execution_root,
        )
        run_id = step_run_id(
            namespace=self._resources.domain.namespace,
            tenant_id=execution.tenant_id,
            execution_id=execution.execution_id,
            segment_sequence=execution.agent_run_sequence,
        )
        conversation_id = step_conversation_id(
            namespace=self._resources.domain.namespace,
            tenant_id=execution.tenant_id,
            execution_id=execution.execution_id,
        )
        try:
            result = await runner.run(
                request.prompt,
                [],
                conversation_id,
                step_store=self._resources.steps,
                step_run_id=run_id,
                segment_sequence=execution.agent_run_sequence,
                memory_namespace=execution.memory_namespace,
                memory_store=None
                if execution.memory_namespace is None
                else RuntimeMemoryStore(
                    self._resources.domain,
                    tenant_id=execution.tenant_id,
                    namespace=execution.memory_namespace,
                ),
                capability_context=CapabilityRuntimeContext(
                    request.principal,
                    ResourceRef(ResourceKind.EXECUTION, execution.execution_id, execution.tenant_id),
                ),
                on_event=self._event_handler,
            )
            await self._commit_success(binding, execution, result)
            return result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._commit_failure(binding, execution, error)
            _logger.error("binding execution failed: execution=%s", execution.execution_id, exc_info=True)
            raise
        finally:
            self._tasks.pop(execution.execution_id, None)

    async def _commit_success(self, binding: AgentBinding, execution: ExecutionRecord, result: WorkspaceAgentResult) -> None:
        now = datetime.now(timezone.utc)
        blob = await self._resources.domain.blobs.put_bytes(tenant_id=execution.tenant_id, data=result.output.encode("utf-8"))
        current = await self._resources.domain.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        terminal = _terminal_record(current, ExecutionStatus.SUCCEEDED, now, result_ref=blob.digest, result_digest=blob.digest)
        identity = await _terminal_idempotency_for(self._resources, current, IdempotencyStatus.COMPLETED, blob.digest, None)
        await self._resources.domain.results.commit_terminal(
            ExecutionTerminalCommit(
                expected_revision=current.revision,
                expected_event_sequence=current.event_sequence,
                execution=terminal,
                result=ResultRecord(
                    current.execution_id,
                    current.tenant_id,
                    ExecutionStatus.SUCCEEDED,
                    binding.spec.output_schema,
                    binding.spec.output_schema_revision,
                    binding.manifest.output_schema_fingerprint,
                    blob.digest,
                    blob.digest,
                    StopReason.END_TURN,
                    0,
                    0,
                    0,
                    now,
                ),
                terminal_event_type=ExecutionEventType.EXECUTION_SUCCEEDED,
                terminal_event_payload={"run_id": result.run_id},
                idempotency=identity,
            )
        )

    async def _commit_failure(self, binding: AgentBinding, execution: ExecutionRecord, error: Exception) -> None:
        current = await self._resources.domain.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
        if current is None or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            return
        now = datetime.now(timezone.utc)
        error_code = ErrorCode.OUTPUT_VALIDATION_FAILED.value if isinstance(error, ValidationError) else error.code.value if isinstance(error, AIError) else ErrorCode.EXECUTION_FAILED.value
        terminal = _terminal_record(current, ExecutionStatus.FAILED, now, error_code=error_code)
        identity = await _terminal_idempotency_for(self._resources, current, IdempotencyStatus.FAILED, None, error_code)
        await self._resources.domain.results.commit_terminal(
            ExecutionTerminalCommit(
                expected_revision=current.revision,
                expected_event_sequence=current.event_sequence,
                execution=terminal,
                result=ResultRecord(
                    current.execution_id,
                    current.tenant_id,
                    ExecutionStatus.FAILED,
                    binding.spec.output_schema,
                    binding.spec.output_schema_revision,
                    binding.manifest.output_schema_fingerprint,
                    None,
                    None,
                    StopReason.OUTPUT_VALIDATION_FAILED if error_code == ErrorCode.OUTPUT_VALIDATION_FAILED.value else StopReason.ERROR,
                    0,
                    0,
                    0,
                    now,
                ),
                terminal_event_type=ExecutionEventType.EXECUTION_FAILED,
                terminal_event_payload={"error_code": error_code},
                idempotency=identity,
            )
        )


def _terminal_record(
    record: ExecutionRecord,
    status: ExecutionStatus,
    now: datetime,
    *,
    result_ref: "str | None" = None,
    result_digest: "str | None" = None,
    error_code: "str | None" = None,
    safe_error_details: "dict[str, JsonValue] | None" = None,
) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=record.execution_id,
        tenant_id=record.tenant_id,
        session_id=record.session_id,
        binding_digest=record.binding_digest,
        parent_execution_id=record.parent_execution_id,
        root_execution_id=record.root_execution_id,
        source_execution_id=record.source_execution_id,
        base_execution_id=record.base_execution_id,
        lineage_kind=record.lineage_kind,
        status=status,
        revision=record.revision + 1,
        event_sequence=record.event_sequence + 1,
        agent_run_sequence=record.agent_run_sequence,
        result_ref=result_ref,
        result_digest=result_digest,
        error_code=error_code,
        safe_error_details={} if safe_error_details is None else safe_error_details,
        created_at=record.created_at,
        updated_at=now,
        memory_namespace=record.memory_namespace,
    )


async def _terminal_idempotency_for(
    resources: "RuntimeResources",
    execution: ExecutionRecord,
    status: IdempotencyStatus,
    result_digest: "str | None",
    error_code: "str | None",
) -> IdempotencyTerminalUpdate | None:
    records = await resources.domain.idempotency.list_by_execution(execution.execution_id, tenant_id=execution.tenant_id)
    if len(records) > 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not records:
        return None
    identity = records[0]
    return IdempotencyTerminalUpdate(identity.scope, identity.key_hash, identity.status, status, identity.request_digest, result_digest, error_code)


__all__ = ["BindingExecutionLauncher"]
