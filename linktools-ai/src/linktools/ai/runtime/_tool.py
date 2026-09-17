#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned tool authorization and durable operation contracts."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import TypeAdapter
from linktools.core import environ
from pydantic_ai.messages import (
    ModelRequest,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.tools import RunContext as PydanticRunContext, ToolDefinition

from ..core import (
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
    PayloadPolicy,
    StoredPayload,
    payload_fits_inline,
)
from ._message import decode_model_messages, encode_model_messages
from ._object import RuntimeObjectKeyFactory, put_runtime_object, read_runtime_object
from .state import RuntimeDomain
from .state._contracts import ToolOperationAdmission
from .state._durability import (
    CommitObservation,
    DurableCommitState,
    run_durable_commit,
)
from .state._contracts import (
    ToolOperationRecord,
)

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


class ToolOperationBridge(Protocol):
    async def begin(
        self,
        ctx: PydanticRunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision: ...

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision: ...

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


class ToolStateRepository(Protocol):
    async def admit(self, request: ToolOperationAdmission) -> ToolOperationRecord: ...
    async def reserve(self, record: ToolOperationRecord) -> ToolOperationRecord: ...
    async def get_operation(
        self, tool_operation_id: str, *, tenant_id: str
    ) -> "ToolOperationRecord | None": ...
    async def list_by_execution(
        self, execution_id: str, *, tenant_id: str
    ) -> tuple[ToolOperationRecord, ...]: ...
    async def claim(
        self, tool_operation_id: str, *, tenant_id: str, owner: str, lease_seconds: int
    ) -> ToolOperationRecord: ...
    async def renew(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        lease_seconds: int,
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
    async def has_by_step_run(self, step_run_id: str, *, tenant_id: str) -> bool: ...


class _ToolOperationRuntimeRepository(Protocol):
    async def admit(self, request: ToolOperationAdmission) -> ToolOperationRecord: ...

    async def has_by_step_run(self, step_run_id: str, *, tenant_id: str) -> bool: ...

    async def list_by_execution(
        self, execution_id: str, *, tenant_id: str
    ) -> tuple[ToolOperationRecord, ...]: ...

    async def existing_call_ids(
        self,
        step_run_id: str,
        tool_call_ids: Sequence[str],
        *,
        tenant_id: str,
    ) -> frozenset[str]: ...

    async def get_by_call(
        self,
        step_run_id: str,
        tool_call_id: str,
        *,
        tenant_id: str,
    ) -> "ToolOperationRecord | None": ...

    async def list_by_step_run(
        self,
        step_run_id: str,
        *,
        tenant_id: str,
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

    async def renew(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        lease_seconds: int,
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
    ) -> ToolOperationRecord: ...

    async def commit_tool_terminal(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: "StoredPayload | None" = None,
        error_code: "str | None" = None,
        error_payload: "StoredPayload | None" = None,
    ) -> ToolOperationRecord: ...

    async def commit_tool_deferred(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
    ) -> ToolOperationRecord: ...


class RuntimeToolOperationBridge:
    """Coordinate one worker's validated tool calls with ToolOperation state."""

    def __init__(
        self,
        repository: _ToolOperationRuntimeRepository,
        recovery_objects: ObjectStore,
        *,
        namespace: str,
        tenant_id: str,
        execution_id: str,
        step_run_id: str,
        binding_digest: str,
        owner: str,
        background_tasks: "set[asyncio.Task[object]]",
        payload_policy: PayloadPolicy,
        recovery_step_run_id: "str | None" = None,
        terminal_commands: "_ToolTerminalCommands | None" = None,
    ) -> None:
        self._repository = repository
        self._recovery_objects = recovery_objects
        self._object_keys = RuntimeObjectKeyFactory(namespace)
        self._tenant_id = validate_tenant_id(tenant_id)
        self._execution_id = validate_resource_id(execution_id)
        self._step_run_id = step_run_id
        self._binding_digest = binding_digest
        self._owner = owner
        self._background_tasks = background_tasks
        self._payload_policy = payload_policy
        self._recovery_step_run_id = recovery_step_run_id
        self._terminal_commands = terminal_commands
        self._decisions: dict[tuple[str, str], ToolOperationDecision] = {}
        self._decision_fingerprints: dict[tuple[str, str], tuple[str, str]] = {}
        self._lease_seconds = 60

    async def begin(
        self,
        ctx: PydanticRunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> "ToolOperationDecision":
        if not isinstance(replay_safe, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        key = (self._run_id(ctx), call.tool_call_id)
        portable_args = _portable_arguments(args)
        arguments_digest = canonical_sha256(portable_args)
        fingerprint = (tool_def.name, arguments_digest)
        prior = self._decisions.get(key)
        if prior is not None:
            if (
                prior.replay_safe is not replay_safe
                or self._decision_fingerprints.get(key) != fingerprint
            ):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return prior
        arguments_payload = await self._arguments_payload(portable_args)
        replay_step_run_id = self._recovery_step_run_id or self._run_id(ctx)
        operation_id = canonical_sha256(
            {
                "tenant_id": self._tenant_id,
                "execution_id": self._execution_id,
                "step_run_id": replay_step_run_id,
                "tool_call_id": call.tool_call_id,
                "tool_name": tool_def.name,
                "arguments_digest": arguments_digest,
                "binding_digest": self._binding_digest,
            }
        )
        admission = ToolOperationAdmission(
            tenant_id=self._tenant_id,
            execution_id=self._execution_id,
            tool_operation_id=operation_id,
            step_run_id=self._run_id(ctx),
            recovery_step_run_id=self._recovery_step_run_id,
            tool_call_id=call.tool_call_id,
            idempotency_key_digest=canonical_sha256(
                {
                    "execution_id": self._execution_id,
                    "step_run_id": replay_step_run_id,
                    "tool_call_id": call.tool_call_id,
                    "tool_name": tool_def.name,
                    "arguments_digest": arguments_digest,
                }
            ),
            tool_name=tool_def.name,
            arguments_digest=arguments_digest,
            binding_digest=self._binding_digest,
            replay_safe=replay_safe,
            owner=self._owner,
            lease_seconds=self._lease_seconds,
            arguments_payload=arguments_payload,
        )
        if self._terminal_commands is not None:
            existing = await self._terminal_commands.commit_tool_admission(admission)
        else:
            existing = await self._repository.admit(admission)
        if existing.tool_call_id != call.tool_call_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        decision = await self._decision_from_record(existing, replay_safe)
        self._decisions[key] = decision
        self._decision_fingerprints[key] = fingerprint
        _logger.debug(
            "tool operation admitted: execution=%s run=%s tool=%s call=%s operation=%s status=%s",
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

    async def existing_call_ids(self, tool_call_ids: Sequence[str]) -> frozenset[str]:
        return await self._repository.existing_call_ids(
            self._step_run_id,
            tool_call_ids,
            tenant_id=self._tenant_id,
        )

    async def list_operations(self) -> tuple[ToolOperationRecord, ...]:
        return await self._repository.list_by_step_run(
            self._step_run_id,
            tenant_id=self._tenant_id,
        )

    async def _decision_from_record(
        self,
        existing: ToolOperationRecord,
        replay_safe: bool,
    ) -> "ToolOperationDecision":
        if existing.execution_id != self._execution_id or existing.replay_safe is not replay_safe:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
            )
        if existing.status is ToolOperationStatus.FAILED:
            return ToolOperationDecision(
                existing.tool_operation_id,
                self._owner,
                existing.fence,
                replay_safe,
                cached_error=await self._decode_error(existing),
            )
        if existing.status is ToolOperationStatus.EFFECT_UNKNOWN:
            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
        if existing.status is not ToolOperationStatus.CLAIMED:
            raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
        return ToolOperationDecision(
            existing.tool_operation_id, self._owner, existing.fence, replay_safe
        )

    async def renew(
        self,
        decision: "ToolOperationDecision",
    ) -> "ToolOperationDecision":
        try:
            record = await self._repository.renew(
                decision.operation_id,
                tenant_id=self._tenant_id,
                owner=self._owner,
                fence=decision.fence,
                lease_seconds=self._lease_seconds,
            )
        except AIError as error:
            _logger.warning(
                "tool operation heartbeat lost: execution=%s operation=%s code=%s",
                self._execution_id,
                decision.operation_id,
                error.code.value,
            )
            raise
        return _decision_type(decision, fence=record.fence)

    async def complete(
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

        async def finish() -> ToolOperationRecord:
            if self._terminal_commands is not None:
                return await self._terminal_commands.commit_tool_terminal(
                    decision.operation_id,
                    tenant_id=self._tenant_id,
                    owner=self._owner,
                    fence=decision.fence,
                    result_payload=payload,
                )
            return await self._repository.complete_payload(
                decision.operation_id,
                tenant_id=self._tenant_id,
                owner=self._owner,
                fence=decision.fence,
                result_payload=payload,
            )

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

        async def finish() -> ToolOperationRecord:
            if self._terminal_commands is not None:
                return await self._terminal_commands.commit_tool_terminal(
                    decision.operation_id,
                    tenant_id=self._tenant_id,
                    owner=self._owner,
                    fence=decision.fence,
                    error_code=code,
                    error_payload=payload,
                )
            return await self._repository.fail_payload(
                decision.operation_id,
                tenant_id=self._tenant_id,
                owner=self._owner,
                fence=decision.fence,
                error_code=code,
                error_payload=payload,
            )

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

        async def finish() -> ToolOperationRecord:
            return await self._repository.mark_effect_unknown(
                decision.operation_id,
                tenant_id=self._tenant_id,
                owner=self._owner,
                fence=decision.fence,
                error_code=code,
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
                    or observed.lease_expires_at is not None
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
        if record.error_payload is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = await self._payload_json(record.error_payload)
        if not isinstance(value, dict):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        version = value.get("version")
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version != 1
        ):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        if set(value) != {"version", "kind", "message"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        kind = value.get("kind")
        message = value.get("message")
        if not isinstance(kind, str) or not isinstance(message, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if kind == "tool_call_rejected":
            if record.error_code != ErrorCode.TOOL_RETRY_REQUIRED.value:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                return ToolCallRetry(message)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if kind == "tool_call_failed":
            if record.error_code != ErrorCode.TOOL_EXECUTION_FAILED.value:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                return ToolCallFailed(message)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

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
        operation: Callable[[], Awaitable[ToolOperationRecord]],
        decision: "ToolOperationDecision",
        *,
        expected_status: ToolOperationStatus,
        expected_payload: StoredPayload,
        expected_error: "str | None" = None,
    ) -> bool:
        async def readback() -> CommitObservation[ToolOperationRecord]:
            observed = await self._repository.get_operation(
                decision.operation_id,
                tenant_id=self._tenant_id,
            )
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
        inline = StoredPayload.inline_bytes(data)
        if payload_fits_inline(inline, self._payload_policy):
            return inline
        reference = await put_runtime_object(
            self._recovery_objects,
            self._object_keys,
            RuntimeDomain.RECOVERY,
            self._tenant_id,
            data,
        )
        return StoredPayload.object(reference)

    async def _arguments_payload(self, args: dict[str, Any]) -> StoredPayload:
        return await self._payload(canonical_json_bytes(_portable_arguments(args)))

    async def _payload_bytes(self, payload: StoredPayload) -> bytes:
        if payload.kind == "inline":
            value = payload.decode()
            if not isinstance(value, bytes):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return value
        if payload.ref is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await read_runtime_object(self._recovery_objects, payload.ref)

    async def _payload_json(self, payload: StoredPayload) -> object:
        try:
            return json.loads((await self._payload_bytes(payload)).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _run_id(self, ctx: PydanticRunContext[None]) -> str:
        del ctx
        return self._step_run_id


def _portable_arguments(args: dict[str, Any]) -> dict[str, Any]:
    try:
        value = TypeAdapter(object).dump_python(args, mode="json")
        normalized = normalize_json_value(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return normalized


def _decision_type(
    decision: "ToolOperationDecision",
    *,
    fence: int,
) -> "ToolOperationDecision":
    return ToolOperationDecision(
        decision.operation_id,
        decision.owner,
        fence,
        decision.replay_safe,
        cached_result=decision.cached_result,
        has_cached_result=decision.has_cached_result,
        cached_error=decision.cached_error,
    )


__all__ = [
    "RuntimeToolOperationBridge",
    "ToolOperationBridge",
    "ToolOperationAdmission",
    "ToolOperationRecord",
    "ToolStateRepository",
]