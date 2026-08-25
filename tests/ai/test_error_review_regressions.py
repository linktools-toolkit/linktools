#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for the error-contract review closure."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import linktools.ai.agent._capabilities as capabilities_module
import linktools.ai.agent._executor as executor_module
import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._capabilities import AgentRunScope
from linktools.ai.agent._executor import AgentExecutor
from linktools.ai.agent._output import AssistantTextOutput, bind_output
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, OperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._execution import CancelEffectOutcome, DefaultExecutionService
from linktools.ai.runtime.service_api import CancelExecutionRequest
from linktools.ai.runtime.state import ExecutionRecord
from linktools.ai.spec import AgentSpec
from linktools.ai.workspace import trusted_workspace_principal
from pydantic_ai.capabilities import ReinjectSystemPrompt
from pydantic_ai_harness.compaction import DeduplicateFileReads


def _binding_snapshot() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "default"),
        agent_digest="b" * 64,
        output_schema_id=output.schema_id,
        output_schema_revision=output.schema_revision,
        output_schema_fingerprint=output.schema_fingerprint,
        local_runtime_capability_descriptors=(),
        binding_digest="a" * 64,
        global_runtime_capability_descriptors=(),
    )


@pytest.mark.asyncio
async def test_default_platform_composition_keeps_file_read_deduplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StepPersistence:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    monkeypatch.setattr(
        capabilities_module,
        "_RuntimeStepPersistence",
        StepPersistence,
    )
    scope = AgentRunScope(
        root=tmp_path,
        agent_name="agent",
        conversation_id=None,
        step_run_id="run",
        segment_sequence=1,
        memory_scope=None,
        step_store=SimpleNamespace(),
        memory_store=None,
    )

    capabilities = await capabilities_module.compose_platform_capabilities(
        scope,
        model_factory=lambda value: value or "model",
        parent_model="model",
    )

    assert any(isinstance(capability, DeduplicateFileReads) for capability in capabilities)


@pytest.mark.asyncio
async def test_agent_executor_cancellation_is_not_replaced_by_usage_sink_failure(tmp_path: Path) -> None:
    executor = AgentExecutor(execution_root=tmp_path)

    async def cancelled(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise asyncio.CancelledError

    async def usage_sink(_usage: object) -> None:
        raise RuntimeError("usage sink failed")

    async def event_sink(_event: object) -> None:
        del _event

    executor._execute = cancelled  # type: ignore[method-assign]
    binding = SimpleNamespace(
        definition=SimpleNamespace(spec=SimpleNamespace(usage_limits=None))
    )

    with pytest.raises(asyncio.CancelledError):
        await executor.execute(
            binding,  # type: ignore[arg-type]
            "prompt",
            [],
            "conversation",
            step_store=SimpleNamespace(),  # type: ignore[arg-type]
            step_run_id="run",
            segment_sequence=1,
            capability_context=SimpleNamespace(),  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            usage_sink=usage_sink,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_history_system_prompt", [False, True])
async def test_agent_executor_reinjects_only_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_history_system_prompt: bool,
) -> None:
    output_binding = bind_output()
    model = SimpleNamespace(profile={})
    definition = SimpleNamespace(
        digest="b" * 64,
        spec=AgentSpec("agent", 1, "default"),
        model=SimpleNamespace(materialize=lambda: model),
        effective_capabilities=(),
        trusted_tool_classes=(),
        trusted_mcp_selectors=(),
    )
    binding = SimpleNamespace(
        definition=definition,
        output_binding=output_binding,
        output_type=output_binding.runtime_output_type,
    )
    captured: dict[str, object] = {}

    class _StepStore:
        def __init__(self) -> None:
            self._get_run_calls = 0

        async def get_run(self, *, run_id: str) -> object | None:
            del run_id
            self._get_run_calls += 1
            if self._get_run_calls == 1:
                return None
            return SimpleNamespace(conversation_id="conversation")

        async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> object:
            del run_id, include_interrupted
            return object()

        async def list_unresolved_tool_effects(self, *, run_id: str) -> tuple[object, ...]:
            del run_id
            return ()

    class _Result:
        run_id = "run"
        output = AssistantTextOutput(text="ok")

        def all_messages(self) -> list[object]:
            return []

    class _ResultEvent:
        def __init__(self) -> None:
            self.result = _Result()

    class _Events:
        def __init__(self) -> None:
            self._done = False

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        def __aiter__(self) -> "_Events":
            return self

        async def __anext__(self) -> _ResultEvent:
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return _ResultEvent()

    class _Agent:
        def run_stream_events(self, *args: object, **kwargs: object) -> _Events:
            del args
            captured["capabilities"] = kwargs["capabilities"]
            return _Events()

    async def compose_platform(*args: object, **kwargs: object) -> tuple[object, ...]:
        del args, kwargs
        return ()

    monkeypatch.setattr(executor_module, "build_pydantic_agent", lambda *args, **kwargs: _Agent())
    monkeypatch.setattr(executor_module, "compose_platform_capabilities", compose_platform)
    monkeypatch.setattr(executor_module, "AgentRunResultEvent", _ResultEvent)

    executor = AgentExecutor(execution_root=tmp_path)
    await executor.execute(
        binding,  # type: ignore[arg-type]
        "new prompt",
        [],
        "conversation",
        step_store=_StepStore(),  # type: ignore[arg-type]
        step_run_id="run",
        segment_sequence=1,
        capability_context=SimpleNamespace(),  # type: ignore[arg-type]
        event_sink=lambda _event: asyncio.sleep(0),  # type: ignore[arg-type]
        replace_history_system_prompt=replace_history_system_prompt,
    )

    capabilities = captured["capabilities"]
    assert isinstance(capabilities, tuple)
    assert any(isinstance(capability, ReinjectSystemPrompt) for capability in capabilities) is replace_history_system_prompt


@pytest.mark.asyncio
async def test_agent_executor_rejects_non_json_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidOutput(AssistantTextOutput):
        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            del args, kwargs
            return {"value": ("invalid",)}

    model = SimpleNamespace(profile={})
    definition = SimpleNamespace(
        digest="b" * 64,
        spec=AgentSpec("agent", 1, "default"),
        model=SimpleNamespace(materialize=lambda: model),
        effective_capabilities=(),
        trusted_tool_classes=(),
        trusted_mcp_selectors=(),
    )
    binding = SimpleNamespace(
        definition=definition,
        output_type=InvalidOutput,
    )

    class _StepStore:
        def __init__(self) -> None:
            self._get_run_calls = 0

        async def get_run(self, *, run_id: str) -> object | None:
            del run_id
            self._get_run_calls += 1
            if self._get_run_calls == 1:
                return None
            return SimpleNamespace(conversation_id="conversation")

        async def latest_snapshot(
            self,
            *,
            run_id: str,
            include_interrupted: bool = False,
        ) -> object:
            del run_id, include_interrupted
            return object()

        async def list_unresolved_tool_effects(
            self,
            *,
            run_id: str,
        ) -> tuple[object, ...]:
            del run_id
            return ()

    class _Result:
        run_id = "run"
        output = InvalidOutput(text="invalid")

        def all_messages(self) -> list[object]:
            return []

    class _ResultEvent:
        def __init__(self) -> None:
            self.result = _Result()

    class _Events:
        def __init__(self) -> None:
            self._done = False

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        def __aiter__(self) -> "_Events":
            return self

        async def __anext__(self) -> _ResultEvent:
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return _ResultEvent()

    class _Agent:
        def run_stream_events(self, *args: object, **kwargs: object) -> _Events:
            del args, kwargs
            return _Events()

    async def compose_platform(*args: object, **kwargs: object) -> tuple[object, ...]:
        del args, kwargs
        return ()

    monkeypatch.setattr(
        executor_module,
        "build_pydantic_agent",
        lambda *args, **kwargs: _Agent(),
    )
    monkeypatch.setattr(executor_module, "compose_platform_capabilities", compose_platform)
    monkeypatch.setattr(executor_module, "AgentRunResultEvent", _ResultEvent)

    async def event_sink(_event: object) -> None:
        del _event

    executor = AgentExecutor(execution_root=tmp_path)
    with pytest.raises(AIError) as raised:
        await executor.execute(
            binding,  # type: ignore[arg-type]
            "prompt",
            [],
            "conversation",
            step_store=_StepStore(),  # type: ignore[arg-type]
            step_run_id="run",
            segment_sequence=1,
            capability_context=SimpleNamespace(),  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
        )

    assert raised.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED
    assert raised.value.retryable is False
    assert raised.value.safe_details == {}


@pytest.mark.asyncio
async def test_confirmed_cancel_persists_canonical_terminal_error() -> None:
    now = datetime.now(timezone.utc)
    principal = trusted_workspace_principal("tenant")
    execution = ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest="a" * 64,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=1,
        event_sequence=1,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        planning=False,
        thinking=False,
        binding=_binding_snapshot(),
    )
    cancelling = replace(
        execution,
        status=ExecutionStatus.CANCELLING,
        revision=2,
        event_sequence=2,
    )

    operation = SimpleNamespace(
        operation_id="operation",
        status=OperationStatus.PENDING,
    )

    class Operations:
        def __init__(self) -> None:
            self.created = False

        async def get(self, operation_id: str, *, tenant_id: str) -> object | None:
            del operation_id, tenant_id
            return operation if self.created else None

        async def append(self, _value: object) -> object:
            self.created = True
            return operation

    class Executions:
        async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord:
            del execution_id, tenant_id
            return cancelling if operations.created else execution

    class Idempotency:
        async def list_by_resource(
            self,
            *args: object,
            **kwargs: object,
        ) -> tuple[object, ...]:
            del args, kwargs
            return ()

    class Backend:
        async def commit_cancel_checkpoint(self, *args: object, **kwargs: object) -> ExecutionRecord:
            del args, kwargs
            return cancelling

        async def cancel(self, _execution: ExecutionRecord) -> CancelEffectOutcome:
            return CancelEffectOutcome.CONFIRMED

        async def abort_start(self, _execution: ExecutionRecord) -> None:
            raise AssertionError("started execution must not use pending-start cleanup")

    class Committer:
        def __init__(self) -> None:
            self.commit = None

        async def commit_terminal_checkpoint(self, commit: object, *, session_id: str | None) -> object:
            del session_id
            self.commit = commit
            return SimpleNamespace()

    operations = Operations()
    committer = Committer()
    service = object.__new__(DefaultExecutionService)
    service._state = SimpleNamespace(
        operations=operations,
        executions=Executions(),
        idempotency=Idempotency(),
    )
    service._backend = Backend()
    service._subagent_cancellation = None
    service._terminal_committer = committer

    async def load_authorized(*args: object, **kwargs: object) -> ExecutionRecord:
        del args, kwargs
        return execution

    async def resolve_cancel_race(*args: object, **kwargs: object) -> None:
        del args, kwargs

    async def verify_terminal(*args: object, **kwargs: object) -> None:
        del args, kwargs

    service._load_authorized = load_authorized
    service._resolve_cancel_race = resolve_cancel_race
    service._terminal_verifier = verify_terminal

    result = await service._cancel(
        "execution",
        CancelExecutionRequest(
            principal=principal,
            idempotency_key="cancel-review-regression",
        ),
    )

    assert result.cancelled is True
    assert committer.commit is not None
    assert committer.commit.execution.status is ExecutionStatus.CANCELLED
    assert committer.commit.execution.error_code == ErrorCode.EXECUTION_CANCELLED.value
    assert committer.commit.execution.safe_error_details == {}
    assert committer.commit.terminal_event_payload == {
        "error_code": ErrorCode.EXECUTION_CANCELLED.value,
        "safe_error_details": {},
    }
