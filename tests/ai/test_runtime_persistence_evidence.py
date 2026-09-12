#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end evidence for current Runtime persistence contracts."""

import asyncio
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
    UsageMetrics,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.runtime._agent_executor import AgentExecutor
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
from linktools.ai.storage import ObjectRef, StoredPayload
from linktools.ai.workspace import Workspace
from linktools.commands.ai.run import _emit_result
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel
from sqlalchemy.ext.asyncio import create_async_engine
from ._runtime_test_helpers import execution_owner_fields


def _binding_snapshot() -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        agent_spec=AgentSpec("agent", model="default"),
        base_model={"provider": "test", "model": "fixture"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "string"},
    )


def _execution() -> ExecutionRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=1,
        event_sequence=0,
        agent_run_sequence=1,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding_snapshot(),
        **execution_owner_fields(),
    )


def _result(execution_id: str, payload_kind: str, now: datetime) -> ResultRecord:
    output = (
        StoredPayload.inline_json({"text": "result"})
        if payload_kind == "inline"
        else StoredPayload.object(ObjectRef("runtime", "result", "c" * 64, 7))
    )
    return ResultRecord(
        execution_id=execution_id,
        tenant_id="tenant",
        output=output,
        stop_reason=StopReason.END_TURN,
        usage=UsageMetrics(),
        created_at=now,
    )


def test_execution_record_writer_accepts_nested_json_result() -> None:
    execution = _execution()
    result = ResultRecord(
        execution_id=execution.execution_id,
        tenant_id=execution.tenant_id,
        output=StoredPayload.inline_json(
            {
                "findings": [
                    {
                        "trace_id": "trace",
                        "labels": [{"name": "priority", "value": "high"}],
                    }
                ]
            }
        ),
        stop_reason=StopReason.END_TURN,
        usage=UsageMetrics(),
        created_at=execution.updated_at,
    )
    value = replace(execution, result=result)

    payload = _encode_persisted_domain(value)

    assert _decode_enveloped_domain(
        encode_envelope({"type": "execution_record", "payload": payload}),
        ExecutionRecord,
    ) == value


class _PersistenceNestedLabel(BaseModel):
    name: str
    value: str


class _PersistenceNestedFinding(BaseModel):
    trace_id: str
    labels: list[_PersistenceNestedLabel]


class _PersistenceNestedOutput(BaseModel):
    findings: list[_PersistenceNestedFinding]


class _PersistenceTestModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    fingerprint = "a" * 64
    semantic_payload: dict[str, JsonValue] = {
        "provider": "test",
        "model": "test",
    }

    def materialize(self) -> TestModel:
        return TestModel(
            custom_output_args={
                "findings": [
                    {
                        "trace_id": "trace",
                        "labels": [
                            {"name": "priority", "value": "high"},
                        ],
                    }
                ]
            }
        )


class _PersistenceTestModels:
    def snapshot(self) -> "_PersistenceTestModels":
        return self

    def resolve(self, route_id: str) -> _PersistenceTestModelBinding:
        if route_id != "default":
            raise AssertionError(f"unexpected model route: {route_id}")
        return _PersistenceTestModelBinding()

    def restore(
        self,
        payload: dict[str, JsonValue],
        *,
        route_id: "str | None" = None,
    ) -> _PersistenceTestModelBinding:
        if route_id not in {None, "default"}:
            raise AssertionError(f"unexpected model route: {route_id}")
        if dict(payload) != _PersistenceTestModelBinding.semantic_payload:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _PersistenceTestModelBinding()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_terminal_stream_allows_immediate_runtime_close(
    tmp_path: Path,
    backend: str,
) -> None:
    workspace_root = tmp_path / "workspace"
    agent_path = workspace_root / ".linktools" / "agents" / "default"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_bytes(
        AgentSpecCodec().encode(
            AgentSpec("default", model="default", allow_tools=())
        )
    )
    if backend == "filesystem":
        state = RuntimeState.filesystem(tmp_path / "runtime")
    else:
        database = tmp_path / "runtime.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
        await provision_runtime_database(engine)
        await engine.dispose()
        state = RuntimeState.sqlite(database)

    try:
        async with Runtime.open(
            Workspace.load(workspace_root, workspace_id="workspace"),
            models=_PersistenceTestModels(),  # type: ignore[arg-type]
            state=state,
        ) as runtime:
            execution = await runtime.agent("default").start("hello")
            terminal_events = []
            async for item in execution.watch():
                if item.depth != 0:
                    continue
                event = item.event
                if event.event_type in {
                    ExecutionEventType.EXECUTION_SUCCEEDED,
                    ExecutionEventType.EXECUTION_FAILED,
                    ExecutionEventType.EXECUTION_CANCELLED,
                }:
                    terminal_events.append(event)
            assert len(terminal_events) == 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_ai_run_interrupt_closes_and_reopens_sqlite_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    agent_path = workspace_root / ".linktools" / "agents" / "default"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_bytes(
        AgentSpecCodec().encode(
            AgentSpec("default", model="default", allow_tools=())
        )
    )
    database = tmp_path / "runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    await provision_runtime_database(engine)
    await engine.dispose()

    execution_started = asyncio.Event()

    async def blocking_execute(self, scope):
        del self, scope
        execution_started.set()
        await asyncio.Event().wait()
        raise AssertionError("blocked execution unexpectedly completed")

    monkeypatch.setattr(AgentExecutor, "execute", blocking_execute)
    workspace = Workspace.load(workspace_root, workspace_id="workspace")
    state = RuntimeState.sqlite(database)
    try:
        async with Runtime.open(
            workspace,
            models=_PersistenceTestModels(),  # type: ignore[arg-type]
            state=state,
        ) as runtime:
            task = asyncio.create_task(
                _emit_result(
                    runtime,
                    "hello",
                    workspace.workspace_id,
                    workspace.workspace_id,
                    False,
                    False,
                    False,
                )
            )
            await execution_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        await state.close()

    reopened = RuntimeState.sqlite(database)
    try:
        async with Runtime.open(
            workspace,
            models=_PersistenceTestModels(),  # type: ignore[arg-type]
            state=reopened,
        ):
            pass
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_session_runtime_persists_and_reads_terminal_result(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    agent_path = workspace_root / ".linktools" / "agents" / "default"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_bytes(
        AgentSpecCodec().encode(
            AgentSpec("default", model="default", allow_tools=())
        )
    )
    database = tmp_path / "runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    await provision_runtime_database(engine)
    await engine.dispose()
    state = RuntimeState.sqlite(database)

    try:
        async with Runtime.open(
            Workspace.load(workspace_root, workspace_id="workspace"),
            models=_PersistenceTestModels(),  # type: ignore[arg-type]
            state=state,
        ) as runtime:
            created = await runtime.agent("default").create_session("session")
            loaded = await runtime.session.get(
                created.session_id,
                principal=runtime.default_principal,
            )
            assert loaded.session_id == created.session_id

            result = await runtime.agent("default").run(
                "hello",
                output=_PersistenceNestedOutput,
                session_id=created.session_id,
                timeout_seconds=10,
            )
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output_fingerprint is not None

            session_record = await state.conversation.sessions.get(
                created.session_id,
                tenant_id=runtime.tenant_id,
            )
            execution = await state.execution.executions.get(
                result.execution_id,
                tenant_id=runtime.tenant_id,
            )
            persisted_result = await state.execution.executions.get_result(
                result.execution_id,
                tenant_id=runtime.tenant_id,
            )
            assert session_record is not None
            assert session_record.active_execution_id is None
            assert execution is not None
            assert execution.status is ExecutionStatus.SUCCEEDED
            assert execution.result is not None
            assert persisted_result == execution.result
            assert persisted_result.output is not None
            assert persisted_result.output.value == {
                "findings": [
                    {
                        "trace_id": "trace",
                        "labels": [
                            {"name": "priority", "value": "high"},
                        ],
                    }
                ]
            }

            terminal_execution = replace(execution, result=None)
            next_execution = replace(terminal_execution, result=persisted_result)
            _encode_persisted_domain(persisted_result)
            _encode_persisted_domain(next_execution)

            inspected = await runtime.execution.inspect(
                result.execution_id,
                principal=runtime.default_principal,
            )
            waited = await runtime.execution.wait(
                result.execution_id,
                principal=runtime.default_principal,
            )
            assert inspected.status is ExecutionStatus.SUCCEEDED
            assert waited.status is ExecutionStatus.SUCCEEDED
            assert waited.output == persisted_result.output.value
            assert waited.output_fingerprint == result.output_fingerprint
    finally:
        await state.close()
