#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned tool authorization and durable operation contracts."""

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from .._compat import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from linktools.core import environ
from pydantic_ai.exceptions import ModelRetry, ToolRetryError
from pydantic_ai.messages import (
    ModelRequest,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.tools import RunContext, ToolDefinition
from ..core import (
    Principal,
    ResourceRef,
    ToolOperationStatus,
    canonical_sha256,
    validate_lease_owner,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..storage import (
    ObjectStore,
    PayloadPolicy,
    StoredPayload,
    payload_fits_inline,
)
from ._message import decode_model_messages, encode_model_messages
from ._object import RuntimeObjectKeyFactory, put_runtime_object, read_runtime_object
from .state._contracts import ToolOperationAdmission
from .state._durability import (
    CommitObservation,
    DurableCommitState,
    run_durable_commit,
)
from .state._plan import RuntimeDomain

if TYPE_CHECKING:
    from ..agent import ToolOperationDecision

_logger = environ.get_logger("ai.runtime.tool")


@dataclass(frozen=True, slots=True)
class ToolOperationRecord:
    tool_operation_id: str
    tenant_id: str
    step_run_id: str
    tool_call_id: str
    idempotency_key_digest: str
    tool_name: str
    arguments_digest: str
    binding_fingerprint: str
    replay_safe: bool
    status: ToolOperationStatus
    owner: "str | None"
    fence: int
    lease_expires_at: "datetime | None"
    error_code: "str | None"
    created_at: datetime
    updated_at: datetime
    result_payload: "StoredPayload | None" = None
    error_payload: "StoredPayload | None" = None

    def __post_init__(self) -> None:
        try:
            validate_tenant_id(self.tenant_id)
            if self.owner is not None:
                validate_lease_owner(self.owner)
        except AIError as error:
            raise ValueError("tool operation lease identity is invalid") from error


class ToolStateRepository(Protocol):
    async def admit(self, request: ToolOperationAdmission) -> ToolOperationRecord: ...
    async def reserve(self, record: ToolOperationRecord) -> ToolOperationRecord: ...
    async def get_operation(self, tool_operation_id: str, *, tenant_id: str) -> "ToolOperationRecord | None": ...
    async def claim(self, tool_operation_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> ToolOperationRecord: ...
    async def renew(self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, lease_seconds: int) -> ToolOperationRecord: ...
    async def fail(self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, error_code: str) -> ToolOperationRecord: ...


class _ToolOperationRuntimeRepository(Protocol):
    async def admit(self, request: ToolOperationAdmission) -> ToolOperationRecord: ...

    async def get_by_call(
        self,
        step_run_id: str,
        tool_call_id: str,
        *,
        tenant_id: str,
    ) -> ToolOperationRecord | None: ...

    async def mark_effect_unknown(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str | None,
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
        error_payload: StoredPayload | None,
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
        result_payload: StoredPayload | None = None,
        error_code: str | None = None,
        error_payload: StoredPayload | None = None,
    ) -> ToolOperationRecord: ...


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    name: str
    replay_safe: bool = False


class ToolAuthorization(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class AllowAllToolPolicy:
    @property
    def fingerprint(self) -> str:
        return canonical_sha256({"tool_policy": "allow_all", "version": 1})

    async def authorize_tool(
        self,
        principal: Principal,
        execution: ResourceRef,
        tool: ToolDescriptor,
        arguments_digest: str,
    ) -> ToolAuthorization:
        del principal, execution, tool, arguments_digest
        return ToolAuthorization.ALLOW


class ToolPolicy(Protocol):
    @property
    def fingerprint(self) -> str: ...

    async def authorize_tool(
        self,
        principal: Principal,
        execution: ResourceRef,
        tool: ToolDescriptor,
        arguments_digest: str,
    ) -> ToolAuthorization: ...


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
        binding_fingerprint: str,
        owner: str,
        payload_policy: PayloadPolicy,
        recovery_step_run_id: str | None = None,
        terminal_commands: _ToolTerminalCommands | None = None,
    ) -> None:
        self._repository = repository
        self._recovery_objects = recovery_objects
        self._object_keys = RuntimeObjectKeyFactory(namespace)
        self._tenant_id = validate_tenant_id(tenant_id)
        self._execution_id = execution_id
        self._step_run_id = step_run_id
        self._binding_fingerprint = binding_fingerprint
        self._owner = owner
        self._payload_policy = payload_policy
        self._recovery_step_run_id = recovery_step_run_id
        self._terminal_commands = terminal_commands
        self._decisions: dict[tuple[str, str], ToolOperationDecision] = {}
        self._lease_seconds = 60

    async def begin(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> "ToolOperationDecision":
        replay_safe = _replay_safe(tool_def)
        key = (self._run_id(ctx), call.tool_call_id)
        prior = self._decisions.get(key)
        if prior is not None:
            return prior
        arguments_digest = canonical_sha256(args)
        replay_step_run_id = self._recovery_step_run_id or self._run_id(ctx)
        operation_id = canonical_sha256(
            {
                "tenant_id": self._tenant_id,
                "step_run_id": self._run_id(ctx),
                "tool_call_id": call.tool_call_id,
                "tool_name": tool_def.name,
                "arguments_digest": arguments_digest,
                "binding_fingerprint": self._binding_fingerprint,
            }
        )
        admission = ToolOperationAdmission(
            tenant_id=self._tenant_id,
            tool_operation_id=operation_id,
            step_run_id=self._run_id(ctx),
            recovery_step_run_id=self._recovery_step_run_id,
            tool_call_id=call.tool_call_id,
            idempotency_key_digest=canonical_sha256(
                {"step_run_id": replay_step_run_id, "tool_call_id": call.tool_call_id}
            ),
            tool_name=tool_def.name,
            arguments_digest=arguments_digest,
            binding_fingerprint=self._binding_fingerprint,
            replay_safe=replay_safe,
            owner=self._owner,
            lease_seconds=self._lease_seconds,
        )
        if self._terminal_commands is not None:
            existing = await self._terminal_commands.commit_tool_admission(admission)
        else:
            existing = await self._repository.admit(admission)
        decision = await self._decision_from_record(existing, replay_safe)
        self._decisions[key] = decision
        _logger.debug(
            "tool operation admitted: execution=%s run=%s tool=%s call=%s operation=%s status=%s",
            self._execution_id,
            self._run_id(ctx),
            tool_def.name,
            call.tool_call_id,
            decision.operation_id,
            "cached" if decision.has_cached_result or decision.cached_error is not None else "claimed",
        )
        return decision

    async def _decision_from_record(
        self,
        existing: ToolOperationRecord,
        replay_safe: bool,
    ) -> "ToolOperationDecision":
        from ..agent import ToolOperationDecision

        if existing.status is ToolOperationStatus.COMPLETED:
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
        return ToolOperationDecision(existing.tool_operation_id, self._owner, existing.fence, replay_safe)

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
        payload = await self._result_payload(decision, result)

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
        error: BaseException,
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

        return await self._finish_with_readback(
            finish,
            decision,
            expected_status=ToolOperationStatus.FAILED,
            expected_payload=payload,
            expected_error=code,
        )

    async def unknown(
        self,
        decision: "ToolOperationDecision",
        error: BaseException,
    ) -> None:
        code = ErrorCode.TOOL_EFFECT_UNKNOWN.value
        try:
            await self._repository.mark_effect_unknown(
                decision.operation_id,
                tenant_id=self._tenant_id,
                owner=self._owner,
                fence=decision.fence,
                error_code=code,
            )
        except AIError as current_error:
            if current_error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise
            observed = await self._repository.get_operation(
                decision.operation_id,
                tenant_id=self._tenant_id,
            )
            if observed is None or observed.status is not ToolOperationStatus.EFFECT_UNKNOWN:
                raise
        _logger.error(
            "tool operation effect became unknown: execution=%s operation=%s error=%s",
            self._execution_id,
            decision.operation_id,
            type(error).__name__,
        )

    async def _result_payload(
        self,
        decision: "ToolOperationDecision",
        result: Any,
    ) -> StoredPayload:
        message = ModelRequest(
            parts=[ToolReturnPart("runtime", result, tool_call_id=decision.operation_id)],
        )
        data = encode_model_messages((message,))
        return await self._payload(data)

    async def _decode_result(self, record: ToolOperationRecord) -> Any:
        payload = record.result_payload
        if payload is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        messages = decode_model_messages(await self._payload_bytes(payload))
        if len(messages) != 1 or not isinstance(messages[0], ModelRequest) or len(messages[0].parts) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        part = messages[0].parts[0]
        if (
            not isinstance(part, ToolReturnPart)
            or part.tool_call_id != record.tool_operation_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return part.content

    async def _decode_error(self, record: ToolOperationRecord) -> BaseException:
        if record.error_payload is None:
            if record.error_code is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                code = ErrorCode(record.error_code)
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            return AIError(code)
        value = await self._payload_json(record.error_payload)
        if not isinstance(value, dict):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        kind = value.get("kind")
        if kind == "model_retry" and isinstance(value.get("message"), str):
            if record.error_code != ErrorCode.TOOL_RETRY_REQUIRED.value:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return ModelRetry(value["message"])
        if kind == "tool_retry" and isinstance(value.get("content"), str):
            if record.error_code != ErrorCode.TOOL_RETRY_REQUIRED.value:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return ToolRetryError(
                RetryPromptPart(
                    value["content"],
                    tool_call_id=str(value.get("tool_call_id", "")),
                )
            )
        if kind == "error" and isinstance(value.get("code"), str):
            try:
                code = ErrorCode(value["code"])
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if record.error_code != code.value:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            safe_details = value.get("safe_details", {})
            if not isinstance(safe_details, dict):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                return AIError(code, safe_details=safe_details)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _error_payload(self, error: BaseException) -> tuple[str, StoredPayload]:
        if isinstance(error, ModelRetry):
            value = {"kind": "model_retry", "message": error.message}
            return ErrorCode.TOOL_RETRY_REQUIRED.value, await self._json_payload(value)
        if isinstance(error, ToolRetryError):
            retry = error.tool_retry
            value = {
                "kind": "tool_retry",
                "content": str(retry.content),
                "tool_call_id": retry.tool_call_id,
            }
            return ErrorCode.TOOL_RETRY_REQUIRED.value, await self._json_payload(value)
        if isinstance(error, AIError):
            return error.code.value, await self._json_payload(
                {
                    "kind": "error",
                    "code": error.code.value,
                    "safe_details": dict(error.safe_details),
                }
            )
        digest = hashlib.sha256(type(error).__qualname__.encode("utf-8")).hexdigest()
        return ErrorCode.TOOL_EXECUTION_FAILED.value, await self._json_payload(
            {
                "kind": "error",
                "code": ErrorCode.TOOL_EXECUTION_FAILED.value,
                "safe_details": {
                    "error_digest": digest,
                    "phase": "tool_execution",
                },
            }
        )

    async def _finish_with_readback(
        self,
        operation: Callable[[], Awaitable[ToolOperationRecord]],
        decision: "ToolOperationDecision",
        *,
        expected_status: ToolOperationStatus,
        expected_payload: StoredPayload,
        expected_error: str | None = None,
    ) -> bool:
        async def readback() -> CommitObservation[ToolOperationRecord]:
            observed = await self._repository.get_operation(
                decision.operation_id,
                tenant_id=self._tenant_id,
            )
            if observed is None:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if observed.status is expected_status:
                try:
                    self._verify_terminal_payload(
                        observed,
                        expected_status=expected_status,
                        expected_payload=expected_payload,
                        expected_error=expected_error,
                    )
                except AIError as error:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=error,
                    )
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=observed,
                )
            if (
                observed.status is ToolOperationStatus.CLAIMED
                and observed.owner == decision.owner
                and observed.fence == decision.fence
            ):
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            return CommitObservation(DurableCommitState.UNRESOLVED)

        result = await run_durable_commit(operation, readback)
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
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    @staticmethod
    def _verify_terminal_payload(
        record: ToolOperationRecord,
        *,
        expected_status: ToolOperationStatus,
        expected_payload: StoredPayload,
        expected_error: str | None = None,
    ) -> None:
        if expected_status is ToolOperationStatus.COMPLETED and record.result_payload != expected_payload:
            raise AIError(ErrorCode.TOOL_RESULT_CONFLICT)
        if expected_status is ToolOperationStatus.FAILED and (
            record.error_payload != expected_payload or record.error_code != expected_error
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

    async def _json_payload(self, value: dict[str, object]) -> StoredPayload:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return await self._payload(data)

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

    def _run_id(self, ctx: RunContext[None]) -> str:
        del ctx
        return self._step_run_id


def _replay_safe(tool_def: ToolDefinition) -> bool:
    metadata = tool_def.metadata or {}
    value = metadata.get("linktools.ai.replay_safe", False)
    if not isinstance(value, bool):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return value


def _decision_type(
    decision: "ToolOperationDecision",
    *,
    fence: int,
) -> "ToolOperationDecision":
    from ..agent import ToolOperationDecision

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
    "AllowAllToolPolicy",
    "RuntimeToolOperationBridge",
    "ToolAuthorization",
    "ToolDescriptor",
    "ToolOperationAdmission",
    "ToolOperationRecord",
    "ToolPolicy",
    "ToolStateRepository",
]
