#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned tool authorization and durable operation contracts."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import TypeAdapter
from pydantic_ai_harness.step_persistence import RunRecord, StepEvent, StepStore, ToolEffectRecord
from linktools.core import environ
from pydantic_ai.messages import (
    ModelRequest,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.tools import RunContext as PydanticRunContext, ToolDefinition

from ..core import (
    deterministic_id,
    JsonValue,
    ToolOperationStatus,
    canonical_json_bytes,
    canonical_sha256,
    normalize_json_value,
    validate_resource_id,
    validate_tenant_id,
)
from ..capability import ToolCallFailed, ToolCallRetry
from ..errors import AIError, ErrorCode
from ..storage import (
    ObjectStore,
    StoredPayload,
)
from ..model.messages import decode_model_messages, encode_model_messages
from ..persistence.objects import ObjectKeyFactory, put_runtime_object, read_runtime_object
from ..model.attachments import read_attachment_payload, store_attachment_payload
from ..persistence import PersistenceDomain
from ..persistence.state import Observed
from ..persistence.contracts import ToolOperationAdmission
from ..spec import CapabilityRef
from ..persistence.durability import CommitObservation, DurableCommitState, await_owned_task, run_durable_commit
from ..persistence.contracts import ToolOperationRecord, decode_tool_operation_failure

if TYPE_CHECKING:
    from ..execution import ExecutionRecord

_logger = environ.get_logger("ai.runtime.tool")


@dataclass(frozen=True, slots=True)
class ToolOperationDecision:
    """Admission result shared by every runtime-owned tool adapter."""

    operation_id: str
    owner: str
    fence: int
    replay_safe: bool
    cached_result: JsonValue = None
    has_cached_result: bool = False
    cached_error: ToolCallRetry | ToolCallFailed | None = None
    budget_admitted: bool = False
    staged: bool = False
    capability_ref: CapabilityRef | None = None


class PendingToolOperations(Protocol):
    """The existing volatile turn owns these facts until its source boundary."""

    def pending_tool_operations(self) -> tuple[ToolOperationRecord, ...] | None: ...
    async def stage_tool_operation(self, request: ToolOperationAdmission) -> ToolOperationRecord | None: ...
    async def settle_tool_operation(
        self, operation_id: str, *, owner: str, fence: int, status: ToolOperationStatus,
        result_payload: StoredPayload | None = None, error_code: str | None = None,
        error_payload: StoredPayload | None = None,
    ) -> bool: ...
    async def materialize_tool_operations(self) -> None: ...


class ToolOperationBridge(Protocol):
    async def record_tool_effect(self, record: ToolEffectRecord) -> None: ...

    async def get_tool_effect(
        self, *, run_id: str, tool_call_id: str,
    ) -> ToolEffectRecord | None: ...

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]: ...

    @property
    def cancellation_requested(self) -> bool: ...

    async def begin(
        self,
        ctx: PydanticRunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
        *,
        capability_ref: CapabilityRef | None = None,
    ) -> ToolOperationDecision | None: ...


    async def require_active(self) -> None: ...

    async def require_fresh_control(self) -> None: ...

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool: ...

    async def fail(
        self,
        decision: ToolOperationDecision,
        error: ToolCallRetry | ToolCallFailed,
    ) -> bool: ...

    async def unknown(
        self, decision: ToolOperationDecision, error: BaseException
    ) -> None: ...

    async def defer(self, decision: ToolOperationDecision) -> bool: ...

    async def existing_call_ids(
        self,
        tool_call_ids: Sequence[str],
    ) -> frozenset[str]: ...

    async def list_operations(self) -> tuple[ToolOperationRecord, ...]: ...


from ..persistence.contracts import ToolStateRepository


class _ToolOperationRuntimeRepository(Protocol):
    async def observe_operation(self, tool_operation_id: str, *, tenant_id: str) -> Observed[ToolOperationRecord] | None: ...

    async def record_tool_effect(
        self, effect: ToolEffectRecord, *, execution_id: str,
        tool_operation_id: str, binding_digest: str,
        recovery_agent_run_id: str | None = None,
        observed: Observed[ToolOperationRecord] | None = None,
    ) -> Observed[ToolOperationRecord]: ...

    async def get_tool_effect(
        self, *, run_id: str, tool_call_id: str, source_run_id: str | None = None,
    ) -> ToolEffectRecord | None: ...

    async def list_unresolved_tool_effects(
        self, *, execution_id: str, run_id: str, source_run_id: str | None = None,
    ) -> list[ToolEffectRecord]: ...

    async def admit(self, request: ToolOperationAdmission) -> ToolOperationRecord: ...

    async def has_by_agent_run(self, agent_run_id: str, *, tenant_id: str) -> bool: ...

    async def list_by_execution(
        self, execution_id: str, *, tenant_id: str
    ) -> tuple[ToolOperationRecord, ...]: ...

    async def existing_call_ids(
        self,
        agent_run_id: str,
        tool_call_ids: Sequence[str],
        *,
        tenant_id: str,
    ) -> frozenset[str]: ...

    async def get_by_call(
        self,
        agent_run_id: str,
        tool_call_id: str,
        *,
        tenant_id: str,
    ) -> "ToolOperationRecord | None": ...

    async def list_by_agent_run(
        self,
        agent_run_id: str,
        *,
        tenant_id: str,
        execution_id: str | None = None,
        recovery_agent_run_id: str | None = None,
    ) -> tuple[ToolOperationRecord, ...]: ...

    async def mark_effect_unknown(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: "str | None",
    ) -> ToolOperationRecord: ...

    async def complete_payload(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: StoredPayload,
    ) -> ToolOperationRecord: ...

    async def fail_payload(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str,
        error_payload: StoredPayload,
    ) -> ToolOperationRecord: ...

    async def defer(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
    ) -> ToolOperationRecord: ...


class _ToolTerminalCommands(Protocol):
    async def commit_tool_admission(
        self,
        request: ToolOperationAdmission,
        *, observed: Observed[ToolOperationRecord] | None = None,
    ) -> Observed[ToolOperationRecord]: ...

    async def commit_tool_terminal(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        observed: Observed[ToolOperationRecord] | None = None,
        result_payload: "StoredPayload | None" = None,
        error_code: "str | None" = None,
        error_payload: "StoredPayload | None" = None,
    ) -> tuple[Observed[ToolOperationRecord], bool]: ...

    async def commit_tool_deferred(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
    ) -> ToolOperationRecord: ...


class ToolAdmissionBoundary(Protocol):
    async def __call__(
        self, request: ToolOperationAdmission, *,
        observed: Observed[ToolOperationRecord] | None,
        event: StepEvent, known_new: bool,
    ) -> Observed[ToolOperationRecord]: ...


class ToolTerminalBoundary(Protocol):
    async def __call__(
        self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int,
        observed: Observed[ToolOperationRecord] | None, event: StepEvent,
        result_payload: StoredPayload | None = None,
        error_code: str | None = None, error_payload: StoredPayload | None = None,
    ) -> tuple[Observed[ToolOperationRecord], bool]: ...


class ToolUnknownBoundary(Protocol):
    async def __call__(
        self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int,
        error_code: str, event: StepEvent,
    ) -> ToolOperationRecord: ...


class RuntimeToolOperationBridge:
    """Coordinate one worker's validated tool calls with ToolOperation state."""

    def __init__(
        self,
        repository: _ToolOperationRuntimeRepository,
        checkpoint_objects: ObjectStore,
        *,
        namespace: str,
        tenant_id: str,
        execution_id: str,
        agent_run_id: str,
        binding_digest: str,
        owner: str,
        background_tasks: "set[asyncio.Task[object]]",
        recovery_agent_run_id: "str | None" = None,
        terminal_commands: "_ToolTerminalCommands | None" = None,
        producer_generation: int | None = None,
        before_effect: Callable[[], Awaitable[None]] | None = None,
        require_active: Callable[[], Awaitable[None]] | None = None,
        require_fresh_control: Callable[[], Awaitable[None]] | None = None,
        fresh_execution: bool = False,
        budget_scope_id: str | None = None,
        pending_tools: PendingToolOperations | None = None,
    ) -> None:
        self._repository = repository
        self._checkpoint_objects = checkpoint_objects
        self._object_keys = ObjectKeyFactory(namespace)
        self._tenant_id = validate_tenant_id(tenant_id)
        self._execution_id = validate_resource_id(execution_id)
        self._agent_run_id = agent_run_id
        self._binding_digest = binding_digest
        self._owner = owner
        self._background_tasks = background_tasks
        self._recovery_agent_run_id = recovery_agent_run_id
        self._fresh_execution = fresh_execution
        self._terminal_commands = terminal_commands
        self._budget_scope_id = budget_scope_id
        self._pending_tools = pending_tools
        if pending_tools is not None and (budget_scope_id is not None or recovery_agent_run_id is not None):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if budget_scope_id is not None and (terminal_commands is None or producer_generation is None):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        self._producer_generation = producer_generation
        self._before_effect = before_effect
        self._require_active = require_active
        self._require_fresh_control = require_fresh_control
        self._cancellation_requested = False
        self._admission_boundary: ToolAdmissionBoundary | None = None
        self._terminal_boundary: ToolTerminalBoundary | None = None
        self._unknown_boundary: ToolUnknownBoundary | None = None
        self._tool_events: dict[str, StepEvent] = {}
        self._admission_attempts: set[str] = set()
        self._native_step_store: StepStore | None = None
        self._native_run: RunRecord | None = None
        self._prepared_boundaries: tuple[ToolAdmissionBoundary, ToolTerminalBoundary, ToolUnknownBoundary | None] | None = None
        if producer_generation is not None and terminal_commands is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        self._effect_receipts: dict[str, Observed[ToolOperationRecord]] = {}
        self._decisions: dict[tuple[str, str], ToolOperationDecision] = {}
        self._decision_fingerprints: dict[tuple[str, str], tuple[str, str]] = {}

    @property
    def boundary_bound(self) -> bool:
        return self._admission_boundary is not None and self._terminal_boundary is not None

    def confirm_native_run(self, record: RunRecord) -> None:
        """Retain the confirmed native lineage for this run's leaf events."""
        if record.run_id != self._agent_run_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        self._native_run = replace(record, metadata=dict(record.metadata))

    def prepare_boundaries(self, *, admission: ToolAdmissionBoundary, terminal: ToolTerminalBoundary,
                           unknown: ToolUnknownBoundary | None = None) -> None:
        """Offer same-group callbacks until the actual native capability is bound."""
        if self.boundary_bound or self._prepared_boundaries is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._prepared_boundaries = admission, terminal, unknown

    def bind_native_step_store(self, store: StepStore) -> None:
        """Bind native event publication; separate stores use explicit split writes."""
        if self._native_step_store is not None and self._native_step_store is not store:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._native_step_store = store
        if not self.boundary_bound:
            admission, terminal, unknown = self._prepared_boundaries or (self._split_admission, self._split_terminal, None)
            self.bind_boundaries(admission=admission, terminal=terminal, unknown=unknown)
        self._prepared_boundaries = None

    async def _split_admission(
        self, request: ToolOperationAdmission, *, observed: Observed[ToolOperationRecord] | None,
        event: StepEvent, known_new: bool,
    ) -> Observed[ToolOperationRecord]:
        try:
            if self._terminal_commands is not None:
                receipt = await self._terminal_commands.commit_tool_admission(request, observed=observed)
            else:
                value = await self._repository.admit(request)
                receipt = await self._repository.observe_operation(value.tool_operation_id, tenant_id=self._tenant_id)
        except AIError as error:
            if error.code is not ErrorCode.TOOL_EFFECT_UNKNOWN:
                raise
            receipt = await self._repository.observe_operation(request.tool_operation_id, tenant_id=self._tenant_id)
            if receipt is None or receipt.value.status is not ToolOperationStatus.EFFECT_UNKNOWN:
                raise
        if receipt is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._publish_native_event(event)
        return receipt

    async def _split_terminal(
        self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int,
        observed: Observed[ToolOperationRecord] | None, event: StepEvent,
        result_payload: StoredPayload | None = None,
        error_code: str | None = None, error_payload: StoredPayload | None = None,
    ) -> tuple[Observed[ToolOperationRecord], bool]:
        if self._terminal_commands is not None:
            receipt, cancelled = await self._terminal_commands.commit_tool_terminal(
                tool_operation_id, tenant_id=tenant_id, owner=owner, fence=fence, observed=observed,
                result_payload=result_payload, error_code=error_code, error_payload=error_payload,
            )
        else:
            if result_payload is not None:
                await self._repository.complete_payload(tool_operation_id, tenant_id=tenant_id,
                    owner=owner, fence=fence, result_payload=result_payload)
            elif error_code is not None and error_payload is not None:
                await self._repository.fail_payload(tool_operation_id, tenant_id=tenant_id,
                    owner=owner, fence=fence, error_code=error_code, error_payload=error_payload)
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            receipt = await self._repository.observe_operation(tool_operation_id, tenant_id=tenant_id)
            if receipt is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            cancelled = False
        await self._publish_native_event(event)
        return receipt, cancelled

    async def _publish_native_event(self, event: StepEvent, *, readback: bool = False) -> None:
        store = self._native_step_store
        if store is None:
            return
        if readback and any(candidate.idempotency_key == event.idempotency_key
                            for candidate in await store.list_events(run_id=event.run_id)):
            return
        await store.append_event(event)

    def bind_boundaries(
        self, *, admission: ToolAdmissionBoundary, terminal: ToolTerminalBoundary,
        unknown: ToolUnknownBoundary | None = None,
    ) -> None:
        """Bind this turn's sole admission/result writer before any leaf runs."""
        if self._decisions or self._admission_boundary is not None or self._terminal_boundary is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._admission_boundary = admission
        self._terminal_boundary = terminal
        self._unknown_boundary = unknown

    def _terminal_event(
        self, decision: ToolOperationDecision, *, failed: bool = False, error: str | None = None,
    ) -> StepEvent:
        started = self._tool_events.get(decision.operation_id)
        if started is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        kind = "tool_call_failed" if failed else "tool_call_completed"
        return replace(
            started, kind=kind, timestamp=datetime.now(UTC), error=error,
            idempotency_key=deterministic_id("tool", decision.operation_id, started.step_index, decision.fence, kind),
        )

    def _unknown_event(self, operation_id: str, fence: int) -> StepEvent:
        started = self._tool_events.get(operation_id)
        if started is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return replace(started, kind="tool_call_failed", timestamp=datetime.now(UTC),
            error=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
            idempotency_key=deterministic_id("tool", operation_id, started.step_index, fence, "unknown"))

    @property
    def cancellation_requested(self) -> bool:
        return self._cancellation_requested

    async def require_active(self) -> None:
        if self._require_active is not None:
            await self._require_active()

    async def require_fresh_control(self) -> None:
        if self._require_fresh_control is not None:
            await self._require_fresh_control()
        else:
            await self.require_active()

    async def begin(
        self,
        ctx: PydanticRunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
        *,
        capability_ref: CapabilityRef | None = None,
    ) -> "ToolOperationDecision | None":
        if (not isinstance(replay_safe, bool)
                or capability_ref is not None and not isinstance(capability_ref, CapabilityRef)):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self._before_effect is not None and self._pending_tools is None:
            await self._before_effect()
        key = (self._run_id(ctx), call.tool_call_id)
        portable_args = _portable_arguments(args)
        arguments_digest = canonical_sha256(portable_args)
        fingerprint = (tool_def.name, arguments_digest)
        prior = self._decisions.get(key)
        if prior is not None:
            if self._decision_fingerprints.get(key) != fingerprint:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            observed = self._effect_receipts.get(call.tool_call_id)
            receipt = None if observed is None else observed.value
            settled = (prior.has_cached_result or prior.cached_error is not None
                       or receipt is not None and receipt.status in {
                           ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED,
                       })
            original = prior.capability_ref
            requested = capability_ref
            if not settled and not prior.staged and (prior.replay_safe is not replay_safe or original != requested):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if receipt is not None and receipt.status is ToolOperationStatus.CLAIMED:
                if self._budget_scope_id is not None:
                    raise AIError(ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED, safe_details={
                        "scope_id": self._budget_scope_id,
                        "dimension": "tool_calls", "reason": "duplicate_dispatch",
                    })
                return prior
            if (receipt is not None and receipt.status is not ToolOperationStatus.PENDING
                    and self._admission_boundary is None):
                decision = await self._decision_from_record(receipt, replay_safe)
                self._decisions[key] = decision
                return decision
        arguments_payload = await self._arguments_payload(portable_args)
        observed = self._effect_receipts.get(call.tool_call_id)
        receipt = None if observed is None else observed.value
        replay_agent_run_id = None if receipt is None else receipt.agent_run_id
        if replay_agent_run_id is None and self._recovery_agent_run_id is not None:
            source = await self._repository.get_by_call(
                self._recovery_agent_run_id, call.tool_call_id, tenant_id=self._tenant_id,
            )
            if source is not None:
                replay_agent_run_id = source.agent_run_id
        replay_agent_run_id = replay_agent_run_id or self._run_id(ctx)
        operation_id = self._operation_id(replay_agent_run_id, call.tool_call_id)
        admission = ToolOperationAdmission(
            execution_id=self._execution_id,
            tool_operation_id=operation_id,
            agent_run_id=self._run_id(ctx),
            recovery_agent_run_id=(replay_agent_run_id if replay_agent_run_id != self._run_id(ctx) else None),
            tool_call_id=call.tool_call_id,
            tool_name=tool_def.name,
            arguments_digest=arguments_digest,
            binding_digest=self._binding_digest,
            replay_safe=replay_safe,
            owner=self._owner,
            arguments_payload=arguments_payload,
            producer_generation=self._producer_generation,
            capability_ref=capability_ref,
        )
        event = None
        known_new = (operation_id not in self._admission_attempts
                     and observed is None and prior is None and self._recovery_agent_run_id is None)
        self._admission_attempts.add(operation_id)
        if self._pending_tools is not None:
            if self._native_step_store is not None:
                await self._pending_tools.materialize_tool_operations()
            staged = await self._pending_tools.stage_tool_operation(admission)
            if staged is not None:
                decision = replace(await self._decision_from_record(staged, replay_safe), staged=True)
                self._decisions[key] = decision
                self._decision_fingerprints[key] = fingerprint
                return decision
        if self._admission_boundary is not None:
            event = StepEvent(
                run_id=self._agent_run_id, kind="tool_call_started", step_index=ctx.run_step,
                conversation_id=ctx.conversation_id,
                agent_name=(self._native_run.agent_name if self._native_run is not None
                            else None if ctx.agent is None else ctx.agent.name),
                parent_run_id=None if self._native_run is None else self._native_run.parent_run_id,
                metadata={} if self._native_run is None else dict(self._native_run.metadata),
                tool_call_id=call.tool_call_id, tool_name=tool_def.name,
                idempotency_key=deterministic_id("tool", operation_id, ctx.run_step, "started"),
            )
            try:
                confirmed = await self._admission_boundary(
                    admission,
                    observed=(observed if observed is not None
                              and observed.value.agent_run_id == self._run_id(ctx) else None),
                    event=event,
                    known_new=known_new,
                )
            except BaseException:
                self._effect_receipts.pop(call.tool_call_id, None)
                raise
            existing = confirmed.value
        elif self._terminal_commands is not None:
            try:
                confirmed = await self._terminal_commands.commit_tool_admission(
                    admission, observed=(observed if observed is not None
                                         and observed.value.agent_run_id == self._run_id(ctx) else None),
                )
            except BaseException:
                self._effect_receipts.pop(call.tool_call_id, None)
                raise
            existing = confirmed.value
        else:
            existing = await self._repository.admit(admission)
            confirmed = await self._repository.observe_operation(existing.tool_operation_id, tenant_id=self._tenant_id)
            if confirmed is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if existing.tool_call_id != call.tool_call_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._effect_receipts[existing.tool_call_id] = confirmed
        if event is not None:
            self._tool_events[existing.tool_operation_id] = event
        if event is not None and existing.status is ToolOperationStatus.EFFECT_UNKNOWN:
            await self._publish_native_event(self._unknown_event(existing.tool_operation_id, existing.fence))
        decision = await self._decision_from_record(existing, replay_safe)
        if self._admission_boundary is not None and (decision.has_cached_result or decision.cached_error is not None):
            await self._publish_native_event(self._terminal_event(
                decision, failed=decision.cached_error is not None,
                error=None if decision.cached_error is None else decision.cached_error.message,
            ))
        self._decisions[key] = decision
        self._decision_fingerprints[key] = fingerprint
        _logger.debug(
            "tool operation admitted: execution=%s agent_run=%s tool=%s call=%s operation=%s status=%s",
            self._execution_id,
            self._run_id(ctx),
            tool_def.name,
            call.tool_call_id,
            decision.operation_id,
            "cached"
            if decision.has_cached_result or decision.cached_error is not None
            else "claimed",
        )
        return decision

    def _operation_id(self, run_id: str, tool_call_id: str) -> str:
        return canonical_sha256({
            "tenant_id": self._tenant_id,
            "execution_id": self._execution_id,
            "agent_run_id": run_id,
            "tool_call_id": tool_call_id,
        })

    def _effect_source_run(self, run_id: str) -> str:
        if run_id not in {self._agent_run_id, self._recovery_agent_run_id}:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        return self._recovery_agent_run_id or run_id

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        source_run_id = self._effect_source_run(record.run_id)
        observed = self._effect_receipts.get(record.tool_call_id)
        receipt = None if observed is None else observed.value
        if self._admission_boundary is not None:
            if receipt is None or record.status != receipt.to_tool_effect(record.run_id).status:
                raise RuntimeError("tool effect lifecycle belongs to the admitted leaf boundary")
        if receipt is not None and receipt.status in {
            ToolOperationStatus.CLAIMED, ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED,
        }:
            if receipt.tool_name != record.tool_name:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if (
                receipt.effect_started_at is not None
                and record.idempotency_key in {None, receipt.effect_idempotency_key}
                and record.effect_summary in {None, receipt.effect_summary}
            ):
                return
        operation = await self._repository.record_tool_effect(
            record,
            execution_id=self._execution_id,
            tool_operation_id=(receipt.tool_operation_id if receipt is not None
                               else self._operation_id(record.run_id, record.tool_call_id)),
            binding_digest=self._binding_digest,
            recovery_agent_run_id=(receipt.agent_run_id if receipt is not None and receipt.agent_run_id != record.run_id
                                   else source_run_id if source_run_id != record.run_id else None),
            observed=observed,
        )
        self._effect_receipts[record.tool_call_id] = operation

    async def get_tool_effect(
        self, *, run_id: str, tool_call_id: str,
    ) -> ToolEffectRecord | None:
        source_run_id = self._effect_source_run(run_id)
        staged = None if self._pending_tools is None else self._pending_tools.pending_tool_operations()
        if staged is not None:
            operation = next((value for value in staged if value.tool_call_id == tool_call_id), None)
            return None if operation is None else operation.to_tool_effect(run_id)
        observed = self._effect_receipts.get(tool_call_id)
        receipt = None if observed is None else observed.value
        if (
            receipt is not None
            and receipt.status is not ToolOperationStatus.EFFECT_UNKNOWN
            and run_id == self._agent_run_id
        ):
            return receipt.to_tool_effect(run_id)
        return await self._repository.get_tool_effect(
            run_id=run_id, tool_call_id=tool_call_id, source_run_id=source_run_id,
        )

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        source_run_id = self._effect_source_run(run_id)
        staged = None if self._pending_tools is None else self._pending_tools.pending_tool_operations()
        if staged is not None:
            return [effect for value in staged if (effect := value.to_tool_effect(run_id)).status == "started"]
        return await self._repository.list_unresolved_tool_effects(
            execution_id=self._execution_id, run_id=run_id, source_run_id=source_run_id,
        )

    async def existing_call_ids(self, tool_call_ids: Sequence[str]) -> frozenset[str]:
        staged = None if self._pending_tools is None else self._pending_tools.pending_tool_operations()
        if staged is not None:
            return frozenset(value.tool_call_id for value in staged if value.tool_call_id in tool_call_ids)
        known = frozenset(call_id for call_id in tool_call_ids if call_id in self._effect_receipts)
        missing = tuple(call_id for call_id in tool_call_ids if call_id not in known)
        if self._fresh_execution and self._recovery_agent_run_id is None:
            missing = tuple(call_id for call_id in missing
                            if self._operation_id(self._agent_run_id, call_id) in self._admission_attempts)
        if not missing:
            return known
        for run_id in dict.fromkeys((self._agent_run_id, self._recovery_agent_run_id)):
            if run_id is not None:
                known |= await self._repository.existing_call_ids(run_id, missing, tenant_id=self._tenant_id)
                missing = tuple(call_id for call_id in missing if call_id not in known)
                if not missing:
                    break
        return known

    def confirmed_terminal_operations(
        self, *, execution: "ExecutionRecord | None" = None,
    ) -> tuple[ToolOperationRecord, ...] | None:
        """Return the complete fresh turn's immutable outcomes, or require a source read."""
        if not self._fresh_execution or self._recovery_agent_run_id is not None:
            return None
        if execution is not None and (
            execution.execution_id != self._execution_id
            or execution.producer_generation != self._producer_generation
            or execution.producer_claim_id != self._owner
            or execution.agent_run_seq != 1
            or execution.recovery is None
            or execution.recovery.agent_run_id != self._agent_run_id
        ):
            return None
        operations = tuple(receipt.value for receipt in self._effect_receipts.values())
        if ({record.tool_operation_id for record in operations} != self._admission_attempts
                or any(record.status not in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED}
                       for record in operations)):
            return None
        return operations

    async def list_operations(self) -> tuple[ToolOperationRecord, ...]:
        staged = None if self._pending_tools is None else self._pending_tools.pending_tool_operations()
        if staged is not None:
            return staged
        confirmed = self.confirmed_terminal_operations()
        if confirmed is not None:
            return confirmed
        return await self._repository.list_by_agent_run(
            self._agent_run_id,
            tenant_id=self._tenant_id,
            execution_id=self._execution_id if self._recovery_agent_run_id is not None else None,
            recovery_agent_run_id=self._recovery_agent_run_id,
        )

    async def _decision_from_record(
        self,
        existing: ToolOperationRecord,
        replay_safe: bool,
    ) -> "ToolOperationDecision":
        if (existing.execution_id != self._execution_id
                or existing.replay_safe is not replay_safe and existing.status not in {
                    ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED,
                }):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        replay_safe = existing.replay_safe
        if existing.status is ToolOperationStatus.COMPLETED:
            if existing.result_payload is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return ToolOperationDecision(
                existing.tool_operation_id,
                self._owner,
                existing.fence,
                replay_safe,
                cached_result=await self._decode_result(existing),
                has_cached_result=True,
                capability_ref=existing.capability_ref,
            )
        if existing.status is ToolOperationStatus.FAILED:
            return ToolOperationDecision(
                existing.tool_operation_id,
                self._owner,
                existing.fence,
                replay_safe,
                cached_error=await self._decode_error(existing),
                capability_ref=existing.capability_ref,
            )
        if existing.status is ToolOperationStatus.EFFECT_UNKNOWN:
            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
        if existing.status is not ToolOperationStatus.CLAIMED or existing.owner != self._owner:
            raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
        return ToolOperationDecision(
            existing.tool_operation_id, self._owner, existing.fence, replay_safe,
            budget_admitted=self._budget_scope_id is not None,
            capability_ref=existing.capability_ref,
        )

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool:
        # Once a leaf returned, its exact value belongs to settlement even when
        # cancellation arrives while its object payload is being written.
        task = asyncio.create_task(self._complete_result(decision, result))
        settled, error, cancelled = await await_owned_task(task, self._background_tasks)
        if error is not None:
            raise error
        if settled is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return settled or cancelled

    async def _complete_result(
        self,
        decision: "ToolOperationDecision",
        result: Any,
    ) -> bool:
        try:
            payload = await self._result_payload(decision, result)
        except BaseException as error:
            _logger.error(
                "tool result encoding failed after execution: execution=%s operation=%s",
                self._execution_id,
                decision.operation_id,
            )
            await self.unknown(decision, error)
            raise

        if (decision.staged and self._pending_tools is not None
                and await self._pending_tools.settle_tool_operation(
                    decision.operation_id, owner=decision.owner, fence=decision.fence,
                    status=ToolOperationStatus.COMPLETED, result_payload=payload,
                )):
            return False

        async def finish() -> Observed[ToolOperationRecord]:
            if self._terminal_boundary is not None:
                record, cancellation_requested = await self._terminal_boundary(
                    decision.operation_id, tenant_id=self._tenant_id, owner=self._owner, fence=decision.fence,
                    observed=next((receipt for receipt in self._effect_receipts.values()
                                   if receipt.value.tool_operation_id == decision.operation_id), None),
                    event=self._terminal_event(decision), result_payload=payload,
                )
                self._cancellation_requested |= cancellation_requested
                return record
            if self._terminal_commands is not None:
                record, cancellation_requested = await self._terminal_commands.commit_tool_terminal(
                    decision.operation_id,
                    tenant_id=self._tenant_id,
                    owner=self._owner,
                    fence=decision.fence,
                    observed=next((receipt for receipt in self._effect_receipts.values()
                                   if receipt.value.tool_operation_id == decision.operation_id), None),
                    result_payload=payload,
                )
                self._cancellation_requested |= cancellation_requested
                return record
            await self._repository.complete_payload(
                decision.operation_id,
                tenant_id=self._tenant_id,
                owner=self._owner,
                fence=decision.fence,
                result_payload=payload,
            )
            observed = await self._repository.observe_operation(decision.operation_id, tenant_id=self._tenant_id)
            if observed is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return observed

        return await self._finish_with_readback(
            finish,
            decision,
            expected_status=ToolOperationStatus.COMPLETED,
            expected_payload=payload,
        )

    async def fail(
        self,
        decision: "ToolOperationDecision",
        error: ToolCallRetry | ToolCallFailed,
    ) -> bool:
        code, payload = await self._error_payload(error)
        if (decision.staged and self._pending_tools is not None
                and await self._pending_tools.settle_tool_operation(
                    decision.operation_id, owner=decision.owner, fence=decision.fence,
                    status=ToolOperationStatus.FAILED, error_code=code, error_payload=payload,
                )):
            return False

        async def finish() -> Observed[ToolOperationRecord]:
            if self._terminal_boundary is not None:
                record, cancellation_requested = await self._terminal_boundary(
                    decision.operation_id, tenant_id=self._tenant_id, owner=self._owner, fence=decision.fence,
                    observed=next((receipt for receipt in self._effect_receipts.values()
                                   if receipt.value.tool_operation_id == decision.operation_id), None),
                    event=self._terminal_event(decision, failed=True, error=error.message),
                    error_code=code, error_payload=payload,
                )
                self._cancellation_requested |= cancellation_requested
                return record
            if self._terminal_commands is not None:
                record, cancellation_requested = await self._terminal_commands.commit_tool_terminal(
                    decision.operation_id,
                    tenant_id=self._tenant_id,
                    owner=self._owner,
                    fence=decision.fence,
                    observed=next((receipt for receipt in self._effect_receipts.values()
                                   if receipt.value.tool_operation_id == decision.operation_id), None),
                    error_code=code,
                    error_payload=payload,
                )
                self._cancellation_requested |= cancellation_requested
                return record
            await self._repository.fail_payload(
                decision.operation_id,
                tenant_id=self._tenant_id,
                owner=self._owner,
                fence=decision.fence,
                error_code=code,
                error_payload=payload,
            )
            observed = await self._repository.observe_operation(decision.operation_id, tenant_id=self._tenant_id)
            if observed is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return observed

        cancelled = await self._finish_with_readback(
            finish,
            decision,
            expected_status=ToolOperationStatus.FAILED,
            expected_payload=payload,
            expected_error=code,
        )
        _logger.debug(
            "tool operation failed: execution=%s operation=%s code=%s signal=%s",
            self._execution_id,
            decision.operation_id,
            code,
            type(error).__name__,
        )
        return cancelled

    async def unknown(
        self,
        decision: "ToolOperationDecision",
        error: BaseException,
    ) -> None:
        code = ErrorCode.TOOL_EFFECT_UNKNOWN.value
        if decision.staged and self._pending_tools is not None:
            await self._pending_tools.settle_tool_operation(
                decision.operation_id, owner=decision.owner, fence=decision.fence,
                status=ToolOperationStatus.EFFECT_UNKNOWN, error_code=code,
            )
            await self._pending_tools.materialize_tool_operations()

        event = (self._unknown_event(decision.operation_id, decision.fence)
                 if self._admission_boundary is not None else None)

        async def finish() -> ToolOperationRecord:
            if self._unknown_boundary is not None and event is not None:
                try:
                    return await self._unknown_boundary(
                        decision.operation_id, tenant_id=self._tenant_id, owner=decision.owner,
                        fence=decision.fence, error_code=code, event=event,
                    )
                except BaseException:
                    # Native storage failure must not erase already established
                    # uncertainty. Keep the effect fact, then repair its event.
                    await self._repository.mark_effect_unknown(
                        decision.operation_id, tenant_id=self._tenant_id, owner=decision.owner,
                        fence=decision.fence, error_code=code,
                    )
                    raise
            record = await self._repository.mark_effect_unknown(
                decision.operation_id, tenant_id=self._tenant_id, owner=decision.owner,
                fence=decision.fence, error_code=code,
            )
            if event is not None:
                await self._publish_native_event(event)
            return record

        async def readback() -> CommitObservation[ToolOperationRecord]:
            observed = await self._repository.get_operation(
                decision.operation_id,
                tenant_id=self._tenant_id,
            )
            if observed is None:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if observed.execution_id != self._execution_id:
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            if observed.status is ToolOperationStatus.EFFECT_UNKNOWN:
                if (
                    observed.owner != decision.owner
                    or observed.fence != decision.fence
                    or observed.error_code != code
                ):
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
                    )
                if event is not None:
                    store = self._native_step_store
                    present = (store is not None and any(candidate.idempotency_key == event.idempotency_key
                               for candidate in await store.list_events(run_id=event.run_id)))
                    if not present:
                        if self._unknown_boundary is not None:
                            await self._unknown_boundary(
                                decision.operation_id, tenant_id=self._tenant_id, owner=decision.owner,
                                fence=decision.fence, error_code=code, event=event,
                            )
                        else:
                            await self._publish_native_event(event)
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=observed,
                )
            if observed.status is ToolOperationStatus.CLAIMED:
                if (
                    observed.owner == decision.owner
                    and observed.fence == decision.fence
                ):
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
                )
            return CommitObservation(
                DurableCommitState.NOT_COMMITTED,
                error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
            )

        result = await run_durable_commit(
            finish,
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED:
            if result.value is not None:
                self._effect_receipts.pop(result.value.tool_call_id, None)
            _logger.error(
                "tool operation effect became unknown: execution=%s operation=%s error=%s",
                self._execution_id,
                decision.operation_id,
                type(error).__name__,
            )
            if result.cancelled:
                raise asyncio.CancelledError
            raise AIError(
                ErrorCode.TOOL_EFFECT_UNKNOWN,
                safe_details={
                    "execution_id": self._execution_id,
                    "operation_id": decision.operation_id,
                    "phase": "tool_effect",
                },
            ) from error
        if result.state is DurableCommitState.NOT_COMMITTED:
            if result.error is not None:
                raise result.error
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details={
                    "execution_id": self._execution_id,
                    "operation_id": decision.operation_id,
                    "phase": "tool_effect_commit",
                },
            ) from error
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from result.error
        if (
            isinstance(result.error, AIError)
            and result.error.code is ErrorCode.TOOL_OPERATION_CONFLICT
        ):
            raise result.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    async def defer(self, decision: "ToolOperationDecision") -> bool:
        if (decision.staged and self._pending_tools is not None
                and await self._pending_tools.settle_tool_operation(
                    decision.operation_id, owner=decision.owner, fence=decision.fence,
                    status=ToolOperationStatus.PENDING,
                )):
            return False
        async def finish() -> ToolOperationRecord:
            if self._terminal_commands is not None:
                return await self._terminal_commands.commit_tool_deferred(
                    decision.operation_id,
                    tenant_id=self._tenant_id,
                    owner=decision.owner,
                    fence=decision.fence,
                )
            return await self._repository.defer(
                decision.operation_id,
                tenant_id=self._tenant_id,
                owner=decision.owner,
                fence=decision.fence,
            )

        async def readback() -> CommitObservation[ToolOperationRecord]:
            observed = await self._repository.get_operation(
                decision.operation_id,
                tenant_id=self._tenant_id,
            )
            if observed is None:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if observed.execution_id != self._execution_id:
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            if observed.status is ToolOperationStatus.PENDING:
                if (
                    observed.owner is not None
                    or observed.fence != decision.fence
                ):
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=observed,
                )
            if observed.status is ToolOperationStatus.CLAIMED:
                if (
                    observed.owner == decision.owner
                    and observed.fence == decision.fence
                ):
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
                )
            return CommitObservation(
                DurableCommitState.NOT_COMMITTED,
                error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
            )

        result = await run_durable_commit(
            finish,
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED:
            if result.value is not None:
                self._effect_receipts.pop(result.value.tool_call_id, None)
            _logger.debug(
                "tool operation deferred: execution=%s operation=%s",
                self._execution_id,
                decision.operation_id,
            )
            return result.cancelled
        if result.state is DurableCommitState.NOT_COMMITTED:
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "tool deferred commit left partial durable state",
            ) from result.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    async def _result_payload(
        self,
        decision: "ToolOperationDecision",
        result: Any,
    ) -> StoredPayload:
        message = ModelRequest(
            parts=[
                ToolReturnPart("runtime", result, tool_call_id=decision.operation_id)
            ],
        )
        data = encode_model_messages((message,))
        return await self._payload(data)

    async def _decode_result(self, record: ToolOperationRecord) -> Any:
        payload = record.result_payload
        if payload is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        messages = decode_model_messages(await self._payload_bytes(payload))
        if (
            len(messages) != 1
            or not isinstance(messages[0], ModelRequest)
            or len(messages[0].parts) != 1
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        part = messages[0].parts[0]
        if (
            not isinstance(part, ToolReturnPart)
            or part.tool_call_id != record.tool_operation_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return part.content

    async def _decode_error(
        self,
        record: ToolOperationRecord,
    ) -> ToolCallRetry | ToolCallFailed:
        message = decode_tool_operation_failure(record.error_code, record.error_payload)
        if record.error_code == ErrorCode.TOOL_RETRY_REQUIRED.value:
            return ToolCallRetry(message)
        return ToolCallFailed(message)

    async def _error_payload(
        self,
        error: ToolCallRetry | ToolCallFailed,
    ) -> tuple[str, StoredPayload]:
        if isinstance(error, ToolCallRetry):
            code = ErrorCode.TOOL_RETRY_REQUIRED.value
            kind = "tool_call_rejected"
        elif isinstance(error, ToolCallFailed):
            code = ErrorCode.TOOL_EXECUTION_FAILED.value
            kind = "tool_call_failed"
        else:
            raise TypeError("tool failure must be a LinkTools tool signal")
        payload = StoredPayload.inline_bytes(
            canonical_json_bytes(
                {
                    "version": 1,
                    "kind": kind,
                    "message": error.message,
                }
            )
        )
        return code, payload

    async def _finish_with_readback(
        self,
        operation: Callable[[], Awaitable[Observed[ToolOperationRecord]]],
        decision: "ToolOperationDecision",
        *,
        expected_status: ToolOperationStatus,
        expected_payload: StoredPayload,
        expected_error: "str | None" = None,
    ) -> bool:
        async def readback() -> CommitObservation[Observed[ToolOperationRecord]]:
            for call_id, prior in tuple(self._effect_receipts.items()):
                if prior.value.tool_operation_id == decision.operation_id:
                    self._effect_receipts.pop(call_id, None)
            receipt = await self._repository.observe_operation(
                decision.operation_id,
                tenant_id=self._tenant_id,
            )
            observed = None if receipt is None else receipt.value
            if observed is None:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if observed.status is expected_status:
                if observed.owner != decision.owner or observed.fence != decision.fence:
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
                    )
                try:
                    self._verify_terminal_payload(
                        observed,
                        expected_status=expected_status,
                        expected_payload=expected_payload,
                        expected_error=expected_error,
                    )
                except AIError as error:
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=error,
                    )
                if self._admission_boundary is not None:
                    await self._publish_native_event(self._terminal_event(
                        decision, failed=expected_status is ToolOperationStatus.FAILED,
                        error=(None if expected_status is ToolOperationStatus.COMPLETED else
                               decode_tool_operation_failure(observed.error_code, observed.error_payload)),
                    ), readback=True)
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=receipt,
                )
            if observed.status in {ToolOperationStatus.CLAIMED, ToolOperationStatus.EFFECT_UNKNOWN}:
                if (
                    observed.owner == decision.owner
                    and observed.fence == decision.fence
                ):
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
                )
            if observed.status is ToolOperationStatus.EFFECT_UNKNOWN:
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.TOOL_EFFECT_UNKNOWN),
                )
            return CommitObservation(
                DurableCommitState.NOT_COMMITTED,
                error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
            )

        result = await run_durable_commit(
            operation,
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED:
            if result.value is not None:
                self._effect_receipts[result.value.value.tool_call_id] = result.value
            return result.cancelled
        if result.state is DurableCommitState.NOT_COMMITTED:
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "tool terminal commit left partial durable state",
            ) from result.error
        if isinstance(result.error, AIError) and result.error.code in {
            ErrorCode.TOOL_OPERATION_CONFLICT,
            ErrorCode.TOOL_RESULT_CONFLICT,
            ErrorCode.TOOL_EFFECT_UNKNOWN,
        }:
            raise result.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    @staticmethod
    def _verify_terminal_payload(
        record: ToolOperationRecord,
        *,
        expected_status: ToolOperationStatus,
        expected_payload: StoredPayload,
        expected_error: "str | None" = None,
    ) -> None:
        if (
            expected_status is ToolOperationStatus.COMPLETED
            and record.result_payload != expected_payload
        ):
            raise AIError(ErrorCode.TOOL_RESULT_CONFLICT)
        if expected_status is ToolOperationStatus.FAILED and (
            record.error_payload != expected_payload
            or record.error_code != expected_error
        ):
            raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)

    async def _payload(self, data: bytes) -> StoredPayload:
        payload = await store_attachment_payload(
            data, format="model-messages", write_object=lambda body: put_runtime_object(
                self._checkpoint_objects, self._object_keys, PersistenceDomain.CHECKPOINT,
                self._tenant_id, body,
            ),
        )
        return payload if payload is not None else StoredPayload.inline_bytes(data)

    async def _arguments_payload(self, args: dict[str, Any]) -> StoredPayload:
        return StoredPayload.inline_bytes(canonical_json_bytes(_portable_arguments(args)))

    async def _payload_bytes(self, payload: StoredPayload) -> bytes:
        if payload.encoding == "json-attachments":
            return await read_attachment_payload(
                payload, read_object=lambda ref: read_runtime_object(self._checkpoint_objects, ref),
            )
        if payload.kind == "inline":
            value = payload.decode()
            if not isinstance(value, bytes):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return value
        if payload.ref is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await read_runtime_object(self._checkpoint_objects, payload.ref)

    def _run_id(self, ctx: PydanticRunContext[None]) -> str:
        del ctx
        return self._agent_run_id


def _portable_arguments(args: dict[str, Any]) -> dict[str, Any]:
    try:
        value = _portable_value(args)
        normalized = normalize_json_value(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return normalized


def _portable_value(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        normalized = [_portable_value(item) for item in value]
        return [
            item
            for _, item in sorted(
                (canonical_json_bytes(normalize_json_value(item)), item)
                for item in normalized
            )
        ]
    if isinstance(value, list):
        return [_portable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_portable_value(item) for item in value]
    if isinstance(value, Mapping) and all(
        isinstance(key, str) for key in value
    ):
        return {key: _portable_value(item) for key, item in value.items()}
    return TypeAdapter(object).dump_python(value, mode="json")


__all__ = [
    "RuntimeToolOperationBridge",
    "PendingToolOperations",
    "ToolOperationBridge",
    "ToolAdmissionBoundary",
    "ToolTerminalBoundary",
    "ToolUnknownBoundary",
    "ToolOperationAdmission",
    "ToolOperationRecord",
    "ToolStateRepository",
]
