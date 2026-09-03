#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused durable tool-effect control-flow regressions."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import BaseModel

from linktools.ai.core import ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._agent_executor import _RuntimePersistenceBoundary
from linktools.ai.runtime._capabilities import ToolOperationDecision, _RuntimeStepPersistence
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationRecord
from linktools.ai.storage import PayloadPolicy
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.exceptions import (
    CallDeferred,
    ModelRetry,
    SkipToolExecution,
    ToolFailed,
    ToolFailedError,
    ToolRetryError,
)
from pydantic_ai.messages import RetryPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.asyncio


@dataclass
class _Effect:
    run_id: str
    tool_call_id: str
    status: str
    effect_summary: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    idempotency_key: str | None = None


class _StepStore:
    def __init__(self) -> None:
        self.effects: list[_Effect] = []

    async def record_tool_effect(self, effect: Any) -> None:
        self.effects.append(_Effect(effect.run_id, effect.tool_call_id, effect.status, effect.effect_summary))

    async def append_event(self, event: Any) -> None:
        del event

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> Any:
        for effect in reversed(self.effects):
            if effect.run_id == run_id and effect.tool_call_id == tool_call_id:
                return effect
        return None


class _Bridge:
    def __init__(
        self,
        replay_safe: bool,
        *,
        begin_error: BaseException | None = None,
        fail_error: BaseException | None = None,
    ) -> None:
        self.decision = ToolOperationDecision("operation", "owner", 1, replay_safe)
        self.begin_error = begin_error
        self.fail_error = fail_error
        self.calls: list[str] = []

    async def begin(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision:
        del ctx, call, tool_def, args
        assert replay_safe is self.decision.replay_safe
        self.calls.append("begin")
        if self.begin_error is not None:
            raise self.begin_error
        return self.decision

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision:
        return decision

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool:
        del decision, result
        self.calls.append("complete")
        return False

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> bool:
        del decision, error
        self.calls.append("fail")
        if self.fail_error is not None:
            raise self.fail_error
        return False

    async def unknown(self, decision: ToolOperationDecision, error: BaseException) -> None:
        del decision, error
        self.calls.append("unknown")


def _context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id="run")


async def _capability(
    replay_safe: bool,
    *,
    begin_error: BaseException | None = None,
    fail_error: BaseException | None = None,
) -> tuple[_RuntimeStepPersistence, _Bridge, _StepStore, RunContext[None], ToolCallPart, ToolDefinition]:
    bridge = _Bridge(
        replay_safe,
        begin_error=begin_error,
        fail_error=fail_error,
    )
    store = _StepStore()
    capability = _RuntimeStepPersistence(
        tool_operations=bridge,
        store=store,
        agent_name="agent",
        run_id="run",
    )
    context = _context()
    call = ToolCallPart("tool", {}, tool_call_id="call")
    definition = ToolDefinition(name="tool", metadata={"linktools.ai.replay_safe": replay_safe})
    await capability.before_tool_execute(context, call=call, tool_def=definition, args={})
    return capability, bridge, store, context, call, definition


async def _effect_free_capability() -> tuple[
    _RuntimeStepPersistence,
    _Bridge,
    _StepStore,
    RunContext[None],
    ToolCallPart,
    ToolDefinition,
]:
    bridge = _Bridge(True)
    store = _StepStore()
    capability = _RuntimeStepPersistence(
        tool_operations=bridge,
        store=store,
        agent_name="agent",
        run_id="run",
        trusted_tool_classes=(("read_file", "filesystem.read"),),
    )
    context = _context()
    call = ToolCallPart("read_file", {"path": "missing"}, tool_call_id="call")
    definition = ToolDefinition(
        name="read_file",
        capability_id="workspace-filesystem",
    )
    await capability.before_tool_execute(
        context,
        call=call,
        tool_def=definition,
        args={"path": "missing"},
    )
    return capability, bridge, store, context, call, definition


async def test_custom_before_hook_rejection_does_not_start_durable_effect() -> None:
    capability, bridge, store, context, call, definition = await _capability(True)

    class RejectBefore(AbstractCapability[None]):
        async def before_tool_execute(
            self,
            ctx: RunContext[None],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
        ) -> dict[str, Any]:
            del self, ctx, call, tool_def, args
            raise ModelRetry("reject before execution")

    combined = CombinedCapability((_RuntimePersistenceBoundary(capability), RejectBefore()))

    with pytest.raises(ModelRetry):
        await combined.before_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
        )

    assert bridge.calls == []
    assert store.effects == []
    assert not capability._calls


async def test_plan_admission_precedes_custom_before_hook() -> None:
    bridge = _Bridge(True)
    store = _StepStore()
    capability = _RuntimeStepPersistence(
        tool_operations=bridge,
        store=store,
        agent_name="agent",
        run_id="run",
        plan_mode=True,
    )
    context = _context()
    call = ToolCallPart("tool", {}, tool_call_id="call")
    definition = ToolDefinition(name="tool", metadata={"linktools.ai.replay_safe": True})
    entered: list[str] = []

    class SideEffectBefore(AbstractCapability[None]):
        async def before_tool_execute(
            self,
            ctx: RunContext[None],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
        ) -> dict[str, Any]:
            del self, ctx, call, tool_def
            entered.append("custom")
            return args

    combined = CombinedCapability((_RuntimePersistenceBoundary(capability), SideEffectBefore()))

    with pytest.raises(AIError) as raised:
        await combined.before_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
        )

    assert raised.value.code is ErrorCode.CAPABILITY_POLICY_CONFLICT
    assert entered == []
    assert bridge.calls == []
    assert store.effects == []
    assert not capability._calls


async def test_custom_wrap_failure_is_inside_durable_effect_boundary() -> None:
    capability, bridge, store, context, call, definition = await _capability(False)

    class FailingWrap(AbstractCapability[None]):
        async def wrap_tool_execute(
            self,
            ctx: RunContext[None],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            handler: Any,
        ) -> Any:
            del self, ctx, call, tool_def, args, handler
            raise RuntimeError("custom middleware failed before inner handler")

    combined = CombinedCapability((_RuntimePersistenceBoundary(capability), FailingWrap()))
    await combined.before_tool_execute(
        context,
        call=call,
        tool_def=definition,
        args={},
    )

    async def raw_handler(_args: dict[str, Any]) -> None:
        raise AssertionError("custom wrapper must fail before the raw tool")

    with pytest.raises(ToolFailed) as raised:
        await combined.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=raw_handler,
        )

    assert raised.value.message == "TOOL_EFFECT_UNKNOWN: verify side effects before retry"
    assert bridge.calls == ["begin", "unknown"]
    assert [effect.status for effect in store.effects] == ["started"]
    assert not capability._calls


async def test_replay_safe_handler_failure_reports_tool_effect_unknown() -> None:
    capability, bridge, store, context, call, definition = await _capability(True)

    async def handler(_args: dict[str, Any]) -> None:
        raise RuntimeError("tool outcome is unknown")

    with pytest.raises(AIError) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    assert raised.value.safe_details == {"phase": "tool_effect_replay"}
    with pytest.raises(AIError) as propagated:
        await capability.on_tool_execute_error(
            context,
            call=call,
            tool_def=definition,
            args={},
            error=raised.value,
        )
    assert propagated.value is raised.value
    assert bridge.calls == ["begin"]
    assert [effect.status for effect in store.effects] == ["started"]
    assert not capability._calls


async def test_effect_free_handler_failure_is_model_visible() -> None:
    capability, bridge, store, context, call, definition = await _effect_free_capability()

    async def handler(_args: dict[str, Any]) -> None:
        raise RuntimeError("host detail must stay internal")

    with pytest.raises(ToolFailed) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={"path": "missing"},
            handler=handler,
        )

    assert raised.value.message == (
        "TOOL_EXECUTION_FAILED: tool execution failed; adapt and continue"
    )
    assert "host detail" not in raised.value.message
    assert bridge.calls == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]
    assert not capability._calls


async def test_effect_free_ai_error_remains_runtime_failure() -> None:
    capability, bridge, store, context, call, definition = await _effect_free_capability()
    failure = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def handler(_args: dict[str, Any]) -> None:
        raise failure

    with pytest.raises(AIError) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={"path": "missing"},
            handler=handler,
        )
    assert raised.value is failure

    with pytest.raises(AIError) as propagated:
        await capability.on_tool_execute_error(
            context,
            call=call,
            tool_def=definition,
            args={"path": "missing"},
            error=failure,
        )
    assert propagated.value is failure
    assert bridge.calls == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]
    assert not capability._calls


async def test_model_retry_is_prefixed_for_model_feedback() -> None:
    capability, bridge, store, context, call, definition = await _capability(True)

    async def handler(_args: dict[str, Any]) -> None:
        raise ModelRetry("correct the path")

    with pytest.raises(ModelRetry) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert raised.value.message == "TOOL_RETRY_REQUIRED: correct the path"
    assert bridge.calls == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]
    assert not capability._calls


async def test_tool_failed_is_prefixed_for_model_feedback() -> None:
    capability, bridge, store, context, call, definition = await _capability(True)

    async def handler(_args: dict[str, Any]) -> None:
        raise ToolFailed("resource is unavailable")

    with pytest.raises(ToolFailed) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert raised.value.message == "TOOL_EXECUTION_FAILED: resource is unavailable"
    assert bridge.calls == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]
    assert not capability._calls


async def test_structured_tool_failed_error_uses_compact_json() -> None:
    capability, bridge, store, context, call, definition = await _capability(True)

    async def handler(_args: dict[str, Any]) -> None:
        raise ToolFailedError(
            ToolReturnPart(
                "tool",
                {"retryable": False, "reason": "missing"},
                tool_call_id="call",
                outcome="failed",
            )
        )

    with pytest.raises(ToolFailed) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert (
        raised.value.message
        == 'TOOL_EXECUTION_FAILED: {"reason":"missing","retryable":false}'
    )
    assert bridge.calls == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]
    assert not capability._calls


async def test_tool_retry_error_is_prefixed_for_model_feedback() -> None:
    capability, bridge, store, context, call, definition = await _capability(True)

    async def handler(_args: dict[str, Any]) -> None:
        raise ToolRetryError(
            RetryPromptPart(
                "correct the path",
                tool_name="tool",
                tool_call_id="call",
            )
        )

    with pytest.raises(ModelRetry) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert raised.value.message == "TOOL_RETRY_REQUIRED: correct the path"
    assert bridge.calls == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]
    assert not capability._calls


async def test_validation_error_is_prefixed_for_replay_safe_tool() -> None:
    capability, bridge, store, context, call, definition = await _capability(True)

    class Payload(BaseModel):
        value: int

    async def handler(_args: dict[str, Any]) -> None:
        Payload.model_validate({"value": {"invalid": True}})

    with pytest.raises(ModelRetry) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert raised.value.message.startswith("TOOL_RETRY_REQUIRED: [")
    assert '"type":"int_type"' in raised.value.message
    assert bridge.calls == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]
    assert not capability._calls


async def test_historical_unknown_effect_is_model_visible_without_reexecution() -> None:
    unknown = AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
    capability, bridge, store, context, call, definition = await _capability(
        False,
        begin_error=unknown,
    )
    entered = False

    async def handler(_args: dict[str, Any]) -> None:
        nonlocal entered
        entered = True

    with pytest.raises(ToolFailed) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert raised.value.message == "TOOL_EFFECT_UNKNOWN: verify side effects before retry"
    assert entered is False
    assert bridge.calls == ["begin"]
    assert store.effects == []
    assert not capability._calls


async def test_cancellation_keeps_cancellation_control_flow() -> None:
    capability, bridge, store, context, call, definition = await _capability(False)

    async def handler(_args: dict[str, Any]) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert bridge.calls == ["begin", "unknown"]
    assert [effect.status for effect in store.effects] == ["started"]
    assert not capability._calls


async def test_model_tool_error_truncation_preserves_head_and_tail() -> None:
    capability, _bridge, _store, context, call, definition = await _capability(True)
    message = "H" * 4500 + "TAIL"
    full = f"TOOL_RETRY_REQUIRED: {message}"

    async def handler(_args: dict[str, Any]) -> None:
        raise ModelRetry(message)

    with pytest.raises(ModelRetry) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    rendered = raised.value.message
    marker = "...[truncated]..."
    tail_chars = 4096 - 1024 - len(marker)
    assert len(rendered) == 4096
    assert rendered[:1024] == full[:1024]
    assert rendered.count(marker) == 1
    assert rendered[-tail_chars:] == full[-tail_chars:]
    assert not capability._calls


async def test_skip_tool_execution_terminalizes_as_success() -> None:
    capability, bridge, store, context, call, definition = await _capability(True)
    result = {"skipped": True}

    async def handler(_args: dict[str, Any]) -> None:
        raise SkipToolExecution(result)

    with pytest.raises(SkipToolExecution) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert raised.value.result == result
    assert bridge.calls == ["begin", "complete"]
    assert [effect.status for effect in store.effects] == ["started", "completed"]
    assert not capability._calls


async def test_dynamic_deferral_is_explicitly_unsupported_when_effect_is_resolvable() -> None:
    capability, bridge, store, context, call, definition = await _capability(True)

    async def handler(_args: dict[str, Any]) -> None:
        raise CallDeferred({"reason": "later"})

    with pytest.raises(AIError) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert raised.value.code is ErrorCode.CAPABILITY_POLICY_CONFLICT
    assert raised.value.safe_details == {
        "tool_name": "tool",
        "reason": "dynamic_deferred_unsupported",
    }
    with pytest.raises(AIError) as propagated:
        await capability.on_tool_execute_error(
            context,
            call=call,
            tool_def=definition,
            args={},
            error=raised.value,
        )
    assert propagated.value is raised.value
    assert bridge.calls == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]
    assert not capability._calls


async def test_replay_unsafe_deferral_after_handler_entry_fails_closed() -> None:
    capability, bridge, store, context, call, definition = await _capability(False)

    async def handler(_args: dict[str, Any]) -> None:
        raise CallDeferred({"reason": "later"})

    with pytest.raises(ToolFailed) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert raised.value.message == "TOOL_EFFECT_UNKNOWN: verify side effects before retry"
    assert bridge.calls == ["begin", "unknown"]
    assert [effect.status for effect in store.effects] == ["started"]
    assert not capability._calls


async def test_failed_terminal_commit_error_is_not_reclassified_as_tool_effect_unknown() -> None:
    commit_error = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
    capability, bridge, store, context, call, definition = await _capability(True, fail_error=commit_error)

    async def handler(_args: dict[str, Any]) -> None:
        raise ModelRetry("retry")

    with pytest.raises(AIError) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )
    assert raised.value is commit_error

    with pytest.raises(AIError) as propagated:
        await capability.on_tool_execute_error(
            context,
            call=call,
            tool_def=definition,
            args={},
            error=commit_error,
        )
    assert propagated.value is commit_error
    assert bridge.calls == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started"]
    assert not capability._calls


def _runtime_bridge() -> RuntimeToolOperationBridge:
    return RuntimeToolOperationBridge(
        object(),
        object(),
        namespace="namespace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="run",
        binding_digest="binding",
        owner="owner",
        background_tasks=set(),
        payload_policy=PayloadPolicy(),
    )


async def test_tool_failed_error_payload_round_trips_structured_content() -> None:
    bridge = _runtime_bridge()
    failure = ToolFailedError(
        ToolReturnPart(
            "tool",
            {"reason": "missing", "retryable": False},
            tool_call_id="call",
            outcome="failed",
        )
    )
    code, payload = await bridge._error_payload(failure)
    now = datetime.now(timezone.utc)
    record = ToolOperationRecord(
        tool_operation_id="operation",
        tenant_id="tenant",
        step_run_id="run",
        tool_call_id="call",
        idempotency_key_digest="idempotency",
        tool_name="tool",
        arguments_digest="arguments",
        binding_digest="binding",
        replay_safe=True,
        status=ToolOperationStatus.FAILED,
        owner=None,
        fence=1,
        lease_expires_at=None,
        error_code=code,
        created_at=now,
        updated_at=now,
        error_payload=payload,
    )

    restored = await bridge._decode_error(record)

    assert isinstance(restored, ToolFailedError)
    assert restored.tool_failed.tool_name == "tool"
    assert restored.tool_failed.tool_call_id == "call"
    assert restored.tool_failed.outcome == "failed"
    assert restored.tool_failed.content == {"reason": "missing", "retryable": False}


async def test_tool_retry_error_payload_round_trips_retry_part() -> None:
    bridge = _runtime_bridge()
    retry = ToolRetryError(
        RetryPromptPart("correct the path", tool_name="read_file", tool_call_id="call")
    )
    code, payload = await bridge._error_payload(retry)
    now = datetime.now(timezone.utc)
    record = ToolOperationRecord(
        tool_operation_id="operation",
        tenant_id="tenant",
        step_run_id="run",
        tool_call_id="call",
        idempotency_key_digest="idempotency",
        tool_name="read_file",
        arguments_digest="arguments",
        binding_digest="binding",
        replay_safe=True,
        status=ToolOperationStatus.FAILED,
        owner=None,
        fence=1,
        lease_expires_at=None,
        error_code=code,
        created_at=now,
        updated_at=now,
        error_payload=payload,
    )

    restored = await bridge._decode_error(record)

    assert isinstance(restored, ToolRetryError)
    assert restored.tool_retry.tool_name == "read_file"
    assert restored.tool_retry.tool_call_id == "call"
    assert restored.tool_retry.content == "correct the path"
