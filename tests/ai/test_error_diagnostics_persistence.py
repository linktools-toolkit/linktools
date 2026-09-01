#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable and public execution error diagnostic evidence."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    JsonValue,
    StopReason,
    ToolOperationStatus,
    UsageMetrics,
)
from linktools.ai.errors import AIError, ErrorCode, ErrorDiagnostics
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.runtime._agent_executor import _execution_error
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationRecord
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
)
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    ExecutionTerminalCommit,
    ResultRecord,
)
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy
from linktools.ai.workspace import Workspace
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits
from sqlalchemy.ext.asyncio import create_async_engine


class _DiagnosticModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:diagnostics"
    fingerprint = "a" * 64
    semantic_payload: dict[str, JsonValue] = {
        "provider": "test",
        "model": "diagnostics",
    }

    def materialize(self) -> TestModel:
        return TestModel()


class _DiagnosticModels:
    def snapshot(self) -> "_DiagnosticModels":
        return self

    def resolve(self, route_id: str) -> _DiagnosticModelBinding:
        if route_id != "default":
            raise AssertionError(f"unexpected model route: {route_id}")
        return _DiagnosticModelBinding()

    def restore(
        self,
        payload: dict[str, JsonValue],
        *,
        route_id: "str | None" = None,
    ) -> _DiagnosticModelBinding:
        if route_id not in {None, "default"}:
            raise AssertionError(f"unexpected model route: {route_id}")
        if dict(payload) != _DiagnosticModelBinding.semantic_payload:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _DiagnosticModelBinding()


def _workspace(root: Path) -> Workspace:
    agent_path = root / ".linktools" / "agents" / "default"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_bytes(
        AgentSpecCodec().encode(
            AgentSpec("default", model="default", allow_tools=())
        )
    )
    return Workspace.load(root)


def _binding_snapshot() -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("default", model="default"),
        model=dict(_DiagnosticModelBinding.semantic_payload),
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "string"},
        binding_digest="a" * 64,
    )


def _started_execution(now: datetime) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="default",
        session_id=None,
        binding_digest="a" * 64,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding_snapshot(),
    )


def _failed_terminal(
    now: datetime,
    diagnostics: ErrorDiagnostics,
) -> tuple[ExecutionRecord, ResultRecord, ExecutionTerminalCommit]:
    started = _started_execution(now)
    details: dict[str, JsonValue] = {"phase": "agent_execution"}
    terminal = replace(
        started,
        status=ExecutionStatus.FAILED,
        revision=1,
        event_sequence=1,
        error_code=ErrorCode.INTERNAL_ERROR.value,
        safe_error_details=details,
        error_diagnostics=diagnostics,
        updated_at=now,
    )
    result = ResultRecord(
        execution_id=started.execution_id,
        tenant_id=started.tenant_id,
        output=None,
        stop_reason=StopReason.ERROR,
        usage=UsageMetrics(),
        created_at=now,
    )
    diagnostic_payload: dict[str, JsonValue] = {
        "exception_type": diagnostics.exception_type,
        "exception_message": diagnostics.exception_message,
        "cause_digest": diagnostics.cause_digest,
    }
    commit = ExecutionTerminalCommit(
        expected_revision=0,
        expected_event_sequence=0,
        execution=terminal,
        result=result,
        terminal_event_type=ExecutionEventType.EXECUTION_FAILED,
        terminal_event_payload={
            "error_code": ErrorCode.INTERNAL_ERROR.value,
            "safe_error_details": details,
            "error_diagnostics": diagnostic_payload,
        },
    )
    return started, result, commit


async def _durable_state(
    tmp_path: Path,
    backend: str,
) -> tuple[RuntimeState, Path]:
    path = tmp_path / f"runtime-{backend}"
    if backend == "filesystem":
        return RuntimeState.filesystem(path), path
    database = path.with_suffix(".db")
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    await provision_runtime_database(engine)
    await engine.dispose()
    return RuntimeState.sqlite(database), database


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_failed_diagnostics_survive_restart_through_public_result_and_event(
    tmp_path: Path,
    backend: str,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    diagnostics = ErrorDiagnostics.from_exception(
        RuntimeError("provider disconnected")
    )
    now = datetime.now(timezone.utc)
    started, _result, commit = _failed_terminal(now, diagnostics)
    state, durable_path = await _durable_state(tmp_path, backend)
    await state.initialize(namespace=workspace.workspace_id, tenant_id="default")
    try:
        await state.execution.executions.create(started)
        await state.execution.executions.commit_terminal(commit)
    finally:
        await state.close()

    reopened = (
        RuntimeState.filesystem(durable_path)
        if backend == "filesystem"
        else RuntimeState.sqlite(durable_path)
    )
    try:
        async with Runtime.open(
            workspace,
            models=_DiagnosticModels(),  # type: ignore[arg-type]
            state=reopened,
        ) as runtime:
            result = await runtime.execution.result(
                started.execution_id,
                principal=runtime.default_principal,
            )
            events = await runtime.event.list(
                started.execution_id,
                principal=runtime.default_principal,
                limit=100,
            )
            terminal = next(
                event
                for event in events.items
                if event.event_type is ExecutionEventType.EXECUTION_FAILED
            )
            expected_payload = {
                "exception_type": diagnostics.exception_type,
                "exception_message": diagnostics.exception_message,
                "cause_digest": diagnostics.cause_digest,
            }
            assert result.status is ExecutionStatus.FAILED
            assert result.error_code == ErrorCode.INTERNAL_ERROR.value
            assert result.safe_error_details == {"phase": "agent_execution"}
            assert result.error_diagnostics == diagnostics
            assert terminal.payload["error_code"] == result.error_code
            assert terminal.payload["safe_error_details"] == dict(
                result.safe_error_details
            )
            assert terminal.payload["error_diagnostics"] == expected_payload
    finally:
        await reopened.close()


def test_historical_execution_without_diagnostics_defaults_to_none() -> None:
    diagnostics = ErrorDiagnostics.from_exception(RuntimeError("legacy"))
    _started, _result, commit = _failed_terminal(
        datetime.now(timezone.utc),
        diagnostics,
    )
    payload = _encode_persisted_domain(commit.execution)
    payload["fields"].pop("error_diagnostics")
    decoded = _decode_enveloped_domain(
        encode_envelope({"type": "execution_record", "payload": payload}),
        ExecutionRecord,
    )
    assert decoded.error_diagnostics is None
    assert decoded.status is ExecutionStatus.FAILED
    assert decoded.error_code == ErrorCode.INTERNAL_ERROR.value


@pytest.mark.asyncio
async def test_standalone_recovery_finalization_keeps_started_execution() -> None:
    backend = object.__new__(LocalExecutionBackend)
    execution = _started_execution(datetime.now(timezone.utc))
    assert execution.session_id is None
    assert execution.status is ExecutionStatus.STARTED
    assert (
        await backend._claim_session_or_recovery_finalizing(execution)
        is execution
    )


def test_model_timeout_preserves_diagnostics_without_changing_safe_contract() -> None:
    error = ModelHTTPError(
        status_code=408,
        model_name="model",
        body={"secret": "provider body"},
    )
    mapped = _execution_error(
        error,
        usage_limits=UsageLimits(),
        run_usage=RunUsage(),
    )
    assert mapped.code is ErrorCode.MODEL_TIMEOUT
    assert mapped.safe_details == {"model_name": "model", "status_code": 408}
    assert mapped.diagnostics == ErrorDiagnostics.from_exception(error)
    assert "secret" not in mapped.safe_details


def _tool_bridge() -> RuntimeToolOperationBridge:
    return RuntimeToolOperationBridge(
        None,  # type: ignore[arg-type]
        InMemoryObjectStore(),
        namespace="diagnostics",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="run",
        binding_digest="a" * 64,
        owner="worker",
        background_tasks=set(),
        payload_policy=PayloadPolicy(),
    )


def _failed_tool_record(
    *,
    error_code: str,
    error_payload: object,
) -> ToolOperationRecord:
    now = datetime.now(timezone.utc)
    return ToolOperationRecord(
        tool_operation_id="operation",
        tenant_id="tenant",
        step_run_id="run",
        tool_call_id="call",
        idempotency_key_digest="b" * 64,
        tool_name="tool",
        arguments_digest="c" * 64,
        binding_digest="a" * 64,
        replay_safe=True,
        status=ToolOperationStatus.FAILED,
        owner=None,
        fence=1,
        lease_expires_at=None,
        error_code=error_code,
        created_at=now,
        updated_at=now,
        error_payload=error_payload,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_tool_error_replay_preserves_diagnostics_and_safe_details() -> None:
    bridge = _tool_bridge()
    error = RuntimeError("tool provider disconnected")
    code, payload = await bridge._error_payload(error)
    decoded = await bridge._decode_error(
        _failed_tool_record(error_code=code, error_payload=payload)
    )
    assert isinstance(decoded, AIError)
    assert decoded.code is ErrorCode.TOOL_EXECUTION_FAILED
    assert decoded.safe_details["phase"] == "tool_execution"
    assert "tool provider disconnected" not in str(decoded.safe_details)
    assert decoded.diagnostics == ErrorDiagnostics.from_exception(error)
