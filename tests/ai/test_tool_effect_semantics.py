#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable tool-effect semantics across Pydantic AI control-flow outcomes."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest
from linktools.ai.agent._capabilities import (
    PLANNING_TOOL_NAMES,
    AgentRunScope,
    ToolOperationDecision,
    _RuntimeStepPersistence,
    compose_platform_capabilities,
)
from linktools.ai.core import ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationRecord
from linktools.ai.storage import PayloadPolicy
from pydantic import BaseModel, ValidationError
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
from pydantic_ai_harness.planning import Planning

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
        self.effects.append(
            _Effect(
                effect.run_id,
                effect.tool_call_id,
                effect.status,
                effect.effect_summary,
            )
        )

    async def append_event(self, event: Any) -> None:
        del event

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> Any:
        for effect in reversed(self.effects):
            if effect.run_id == run_id and effect.tool_call_id == tool_call_id:
                return effect
        return None


class _ToolBridge:
    def __init__(
        self,
        decision: ToolOperationDecision,
        *,
        fail_error: BaseException | None = None,
    ) -> None:
        self.decision = decision
        self.fail_error = fail_error
        self.calls: list[tuple[str, str]] = []
        self.replay_safe: bool | None = None

    async def begin(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision:
        del ctx, call, tool_def, args
        self.calls.append(("begin", ""))
        self.replay_safe = replay_safe
        assert replay_safe is self.decision.replay_safe
        return self.decision

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision:
        self.calls.append(("renew", decision.operation_id))
        return decision

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool:
        del result
        self.calls.append(("complete", decision.operation_id))
        return False

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> bool:
        del error
        self.calls.append(("fail", decision.operation_id))
        if self.fail_error is not None:
            raise self.fail_error
        return False

    async def unknown(self, decision: ToolOperationDecision, error: BaseException) -> None:
        del error
        self.calls.append(("unknown", decision.operation_id))


def _context(run_id: str = "run") -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id=run_id,
    )


def _call(tool_name: str = "tool", tool_call_id: str = "call") -> ToolCallPart:
    return ToolCallPart(tool_name, {}, tool_call_id=tool_call_id)


def _definition(
    replay_safe: bool,
    *,
    name: str = "tool",
    capability_id: str | None = None,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        capability_id=capability_id,
        metadata={"linktools.ai.replay_safe": replay_safe},
    )


async def _capability(
    replay_safe: bool,
    *,
    fail_error: BaseException | None = None,
) -> tuple[
    _RuntimeStepPersistence,
    _ToolBridge,
    _StepStore,
    RunContext[None],
    ToolCallPart,
    ToolDefinition,
]:
    decision = ToolOperationDecision("operation", "owner", 1, replay_safe)
    bridge = _ToolBridge(decision, fail_error=fail_error)
    store = _StepStore()
    capability = _RuntimeStepPersistence(
        tool_operations=bridge,
        store=store,
        agent_name="agent",
        run_id="run",
    )
    context = _context()
    call = _call()
    definition = _definition(replay_safe)
    await capability.before_tool_execute(
        context,
        call=call,
        tool_def=definition,
        args={},
    )
    return capability, bridge, store, context, call, definition


@pytest.mark.parametrize("replay_safe", [False, True])
async def test_tool_failed_is_known_failure_independent_of_replay_safety(
    replay_safe: bool,
) -> None:
    capability, bridge, store, context, call, definition = await _capability(
        replay_safe
    )

    async def handler(_args: dict[str, Any]) -> None:
        raise ToolFailed("missing resource")

    with pytest.raises(ToolFailed, match="missing resource"):
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert [name for name, _ in bridge.calls] == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]
    assert not capability._calls


@pytest.mark.parametrize("replay_safe", [False, True])
async def test_tool_failed_error_is_known_failure_independent_of_replay_safety(
    replay_safe: bool,
) -> None:
    capability, bridge, store, context, call, definition = await _capability(
        replay_safe
    )
    failure = ToolFailedError(
        ToolReturnPart(
            "tool",
            {"reason": "missing"},
            tool_call_id="call",
            outcome="failed",
        )
    )

    async def handler(_args: dict[str, Any]) -> None:
        raise failure

    with pytest.raises(ToolFailedError) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert raised.value is failure
    assert [name for name, _ in bridge.calls] == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]
    assert not capability._calls


@pytest.mark.parametrize("replay_safe", [False, True])
async def test_tool_retry_error_is_known_failure_independent_of_replay_safety(
    replay_safe: bool,
) -> None:
    capability, bridge, store, context, call, definition = await _capability(
        replay_safe
    )
    retry = ToolRetryError(
        RetryPromptPart(
            "correct the arguments",
            tool_name="tool",
            tool_call_id="call",
        )
    )

    async def handler(_args: dict[str, Any]) -> None:
        raise retry

    with pytest.raises(ToolRetryError) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert raised.value is retry
    assert [name for name, _ in bridge.calls] == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]
    assert not capability._calls


class _ValidatedValue(BaseModel):
    value: int


def _validation_error() -> ValidationError:
    try:
        _ValidatedValue.model_validate({"value": "not-an-int"})
    except ValidationError as error:
        return error
    raise AssertionError("validation error was not raised")


@pytest.mark.parametrize("replay_safe", [False, True])
async def test_validation_error_is_known_failure_before_upstream_retry_mapping(
    replay_safe: bool,
) -> None:
    capability, bridge, store, context, call, definition = await _capability(
        replay_safe
    )
    validation_error = _validation_error()

    async def handler(_args: dict[str, Any]) -> None:
        raise validation_error

    with pytest.raises(ValidationError) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )
    assert raised.value is validation_error

    with pytest.raises(ValidationError) as hook_raised:
        await capability.on_tool_execute_error(
            context,
            call=call,
            tool_def=definition,
            args={},
            error=validation_error,
        )
    assert hook_raised.value is validation_error
    assert [name for name, _ in bridge.calls] == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]
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
    assert [name for name, _ in bridge.calls] == ["begin", "complete"]
    assert [effect.status for effect in store.effects] == ["started", "completed"]
    assert not capability._calls


async def test_replay_safe_deferral_preserves_claimed_operation() -> None:
    capability, bridge, store, context, call, definition = await _capability(True)

    async def handler(_args: dict[str, Any]) -> None:
        raise CallDeferred({"reason": "later"})

    with pytest.raises(CallDeferred):
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert [name for name, _ in bridge.calls] == ["begin"]
    assert [effect.status for effect in store.effects] == ["started"]
    assert not capability._calls


async def test_replay_unsafe_deferral_after_handler_entry_fails_closed() -> None:
    capability, bridge, store, context, call, definition = await _capability(False)

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

    assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    assert [name for name, _ in bridge.calls] == ["begin", "unknown"]
    assert [effect.status for effect in store.effects] == ["started"]


async def test_failed_terminal_commit_error_is_not_reclassified_as_tool_effect_unknown() -> None:
    commit_error = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
    capability, bridge, store, context, call, definition = await _capability(
        False,
        fail_error=commit_error,
    )

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

    with pytest.raises(AIError) as hook_raised:
        await capability.on_tool_execute_error(
            context,
            call=call,
            tool_def=definition,
            args={},
            error=commit_error,
        )
    assert hook_raised.value is commit_error
    assert [name for name, _ in bridge.calls] == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started"]
    assert not capability._calls


async def test_planning_capability_exposes_only_linktools_planning_surface(
    tmp_path: Any,
) -> None:
    scope = AgentRunScope(
        root=tmp_path,
        agent_name="agent",
        conversation_id=None,
        step_run_id="run",
        segment_sequence=1,
        memory_scope=None,
        step_store=_StepStore(),
        memory_store=None,
        platform_tool_names=PLANNING_TOOL_NAMES,
        planning=True,
    )

    capabilities = await compose_platform_capabilities(
        scope,
        model_factory=lambda value: value or "test",
        parent_model="test",
    )
    planning = next(
        capability
        for capability in capabilities
        if isinstance(capability, Planning)
    )

    assert tuple(planning.tools or ()) == PLANNING_TOOL_NAMES


def _runtime_bridge() -> RuntimeToolOperationBridge:
    return RuntimeToolOperationBridge(
        object(),
        object(),
        namespace="namespace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="run",
        binding_fingerprint="binding",
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
        binding_fingerprint="binding",
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
        RetryPromptPart(
            "correct the path",
            tool_name="read_file",
            tool_call_id="call",
        )
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
        binding_fingerprint="binding",
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
