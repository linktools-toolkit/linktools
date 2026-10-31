#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Durable and public execution error diagnostic evidence."""
from ._runtime_test_helpers import model_features

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.agent import AgentBindingContract
from linktools.ai.capability import ToolCallFailed
from linktools.ai.runtime import RuntimeFeatures
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
from linktools.ai.history import History
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence import Persistence
from linktools.ai.runtime._agent_executor import _execution_error
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationRecord
from linktools.ai.persistence.codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
)
from linktools.ai.persistence.contracts import (
    ExecutionRecord,
    ExecutionTerminalCommit,
    ResultRecord,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import FilesystemObjectStore, InMemoryObjectStore, PayloadPolicy, StoredPayload
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits
from sqlalchemy.ext.asyncio import create_async_engine
from ._runtime_test_helpers import execution_owner_fields


class _DiagnosticModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:diagnostics"
    contract: dict[str, JsonValue] = {
        "provider": "test",
        "model": "diagnostics",
    }

    def materialize(self) -> TestModel:
        return TestModel()


class _DiagnosticModels(_DiagnosticModelBinding):
    pass


def _binding_contract() -> AgentBindingContract:
    return AgentBindingContract(
        agent_spec=AgentSpec("default", model="default", tool_retries=3, output_retries=3),
        model_contract=dict(_DiagnosticModelBinding.contract),
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "string"},
    )


def _started_execution(now: datetime) -> ExecutionRecord:
    binding = _binding_contract()
    return ExecutionRecord(
        execution_id="execution",
        session_id=None,
        parent_execution_id=None,
        root_execution_id="execution",
        previous_execution_id=None,
        fork_base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=0,
        event_seq=0,
        agent_run_seq=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=binding,
        **execution_owner_fields("diagnostic prompt"),
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
        event_seq=1,
        error_code=ErrorCode.INTERNAL_ERROR.value,
        safe_error_details=details,
        error_diagnostics=diagnostics,
        updated_at=now,
    )
    result = ResultRecord(
        output=None,
        stop_reason=StopReason.ERROR,
        usage=UsageMetrics(),
        created_at=now,
    )
    diagnostic_payload = diagnostics.to_payload()
    commit = ExecutionTerminalCommit(
        expected_revision=0,
        expected_event_seq=0,
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
) -> tuple[Persistence, Path]:
    path = tmp_path / f"runtime-{backend}"
    if backend == "filesystem":
        return Persistence(PersistenceStorage.filesystem(path)), path
    database = path.with_suffix(".db")
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    await provision_runtime_database(engine)
    await engine.dispose()
    return (
        Persistence(PersistenceStorage.sqlite(
            database,
            object_store=FilesystemObjectStore(tmp_path / "objects"),
        )),
        database,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
@pytest.mark.parametrize("status", (
    ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
))
async def test_terminal_results_survive_restart_through_live_and_read_only_apis(
    tmp_path: Path,
    backend: str,
    status: ExecutionStatus,
) -> None:
    now = datetime.now(timezone.utc)
    if status is ExecutionStatus.FAILED:
        diagnostics = ErrorDiagnostics.from_exception(RuntimeError("provider disconnected"))
        started, _result, commit = _failed_terminal(now, diagnostics)
    else:
        started = _started_execution(now)
        cancelled = status is ExecutionStatus.CANCELLED
        commit = ExecutionTerminalCommit(
            expected_revision=0,
            expected_event_seq=0,
            execution=replace(
                started, status=status, revision=1, event_seq=1,
                error_code=ErrorCode.EXECUTION_CANCELLED.value if cancelled else None,
            ),
            result=ResultRecord(
                output=None if cancelled else StoredPayload.inline_json({"answer": "ok"}),
                stop_reason=StopReason.CANCELLED if cancelled else StopReason.END_TURN,
                usage=UsageMetrics(),
                created_at=now,
            ),
            terminal_event_type=(
                ExecutionEventType.EXECUTION_CANCELLED if cancelled
                else ExecutionEventType.EXECUTION_SUCCEEDED
            ),
            terminal_event_payload={},
        )
    state, durable_path = await _durable_state(tmp_path, backend)
    await state.initialize(namespace="default", tenant_id="default")
    try:
        await state.execution.executions.create(started)
        await state.execution.executions.commit_terminal(commit)
    finally:
        await state.close()

    reopened = (
        PersistenceStorage.filesystem(durable_path)
        if backend == "filesystem"
        else PersistenceStorage.sqlite(
            durable_path,
            object_store=FilesystemObjectStore(tmp_path / "objects"),
        )
    )
    async with Persistence.open('default', storage=reopened) as persistence_212, Runtime.attach(persistence=persistence_212, features=model_features(_DiagnosticModels())) as runtime:
        result = await runtime.executions.result(
            started.execution_id,
            principal=runtime.default_principal,
        )
        events = await runtime.events.list(
            started.execution_id,
            principal=runtime.default_principal,
            limit=100,
        )
        terminal = next(
            event
            for event in events.items
            if event.event_type == commit.terminal_event_type
        )
        assert result.status is status
        assert result.error_code == commit.execution.error_code
        assert result.safe_error_details == commit.execution.safe_error_details
        assert result.error_diagnostics == commit.execution.error_diagnostics
        assert result.output == ({"answer": "ok"} if status is ExecutionStatus.SUCCEEDED else None)
        assert await runtime.history.result(
            started.execution_id,
            principal=runtime.default_principal,
        ) == result
        if status is ExecutionStatus.FAILED:
            summary = await runtime.history.inspect_execution(
                started.execution_id,
                principal=runtime.default_principal,
            )
            assert summary.error_diagnostics is None
            assert terminal.payload["error_code"] == result.error_code
            assert terminal.payload["safe_error_details"] == dict(result.safe_error_details)
            assert terminal.payload["error_diagnostics"] == diagnostics.to_payload()

    history_storage = (
        PersistenceStorage.filesystem(durable_path)
        if backend == "filesystem"
        else PersistenceStorage.sqlite(
            durable_path,
            object_store=FilesystemObjectStore(tmp_path / "objects"),
        )
    )
    async with Persistence.open('default', storage=history_storage, tenant_id="default", read_only=True) as history_persistence_257, History.open(persistence=history_persistence_257) as history:
        assert await history.result(
            started.execution_id,
            principal=runtime.default_principal,
        ) == result
        if status is ExecutionStatus.FAILED:
            summary = await history.inspect_execution(
                started.execution_id,
                principal=runtime.default_principal,
            )
            assert summary.error_diagnostics is None


def test_execution_uses_declared_error_diagnostics_default() -> None:
    diagnostics = ErrorDiagnostics.from_exception(RuntimeError("current"))
    _started, _result, commit = _failed_terminal(
        datetime.now(timezone.utc),
        diagnostics,
    )
    payload = _encode_persisted_domain(commit.execution)
    payload["fields"].pop("error_diagnostics")

    restored = _decode_enveloped_domain(
        encode_envelope({"type": "execution_record", "payload": payload}), ExecutionRecord,
    )
    assert restored == replace(commit.execution, error_diagnostics=None)


def test_execution_uses_declared_started_at_default() -> None:
    now = datetime.now(timezone.utc)
    current = replace(_started_execution(now), started_at=now)
    payload = _encode_persisted_domain(current)
    payload["fields"].pop("started_at")

    restored = _decode_enveloped_domain(
        encode_envelope({"type": "execution_record", "payload": payload}), ExecutionRecord,
    )
    assert restored == replace(current, started_at=None)


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
        agent_run_id="run",
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
        execution_id="execution",
        agent_run_id="run",
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
async def test_tool_error_replay_rejects_non_signal_failure() -> None:
    bridge = _tool_bridge()
    error = RuntimeError("tool provider disconnected")
    with pytest.raises(TypeError):
        await bridge._error_payload(error)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_tool_error_replay_restores_only_linktools_signal() -> None:
    bridge = _tool_bridge()
    signal = ToolCallFailed("tool execution failed")
    code, payload = await bridge._error_payload(signal)
    decoded = await bridge._decode_error(
        _failed_tool_record(error_code=code, error_payload=payload)
    )
    assert isinstance(decoded, ToolCallFailed)
    assert decoded.message == signal.message
