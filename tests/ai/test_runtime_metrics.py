#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime automatic Metrics producer and fail-open regressions."""

import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import (
    InMemoryMetricStore,
    MetricQuery,
    MetricWindow,
    Metrics,
    Observation,
)
from linktools.ai.runtime import Runtime
from linktools.ai.runtime import _metrics as runtime_metrics
from linktools.ai.runtime._metric_capability import _RuntimeMetricCapability
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.task import TaskLease
from linktools.ai.workspace import Workspace
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition


class _TextModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    fingerprint = "d" * 64
    semantic_payload: dict[str, JsonValue] = {"provider": "test", "model": "test"}

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="ok")


class _TextModels:
    def snapshot(self) -> "_TextModels":
        return self

    def resolve(self, route_id: str) -> _TextModelBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _TextModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _TextModelBinding:
        if (
            route_id not in {None, "default"}
            or dict(payload) != _TextModelBinding.semantic_payload
        ):
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _TextModelBinding()


class _CaptureRecorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


class _FailingMetricStore:
    async def put_definition(self, namespace: str, definition: object) -> None:
        del namespace, definition
        raise AssertionError("automatic producers do not define metrics")

    async def get_definition(
        self,
        namespace: str,
        name: str,
        revision: int | None,
    ) -> None:
        del namespace, name, revision
        raise AssertionError("automatic producers do not read definitions")

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        del namespace, observations
        raise RuntimeError("metrics backend unavailable")

    async def scan_observations(
        self,
        *args: object,
        **kwargs: object,
    ) -> tuple[Observation, ...]:
        del args, kwargs
        return ()

    async def prune(self, namespace: str, *, before: datetime) -> int:
        del namespace, before
        return 0


class _BlockingMetricStore(_FailingMetricStore):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.batches: list[tuple[Observation, ...]] = []

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        del namespace
        self.entered.set()
        await self.release.wait()
        self.batches.append(observations)


def _write_default_agent(root: Path) -> None:
    path = root / ".linktools" / "agents" / "default"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        AgentSpecCodec().encode(
            AgentSpec("default", model="default", allow_tools=())
        )
    )


def _observation(observation_id: str) -> Observation:
    return Observation(
        version=1,
        observation_id=observation_id,
        kind="test.runtime.metric",
        occurred_at=datetime.now(timezone.utc),
        source_namespace="workspace",
        tenant_id="default",
        status="SUCCEEDED",
        error_code=None,
        correlation={},
        dimensions={},
        measurements=(),
    )


@pytest.mark.asyncio
async def test_runtime_projects_model_agent_and_execution_metrics(tmp_path: Path) -> None:
    _write_default_agent(tmp_path)
    store = InMemoryMetricStore()
    metrics = Metrics.from_store(store, namespace="runtime-metrics")
    start = datetime.now(timezone.utc) - timedelta(seconds=1)
    secret = "TOP_SECRET_PROMPT_VALUE"

    async with Runtime.open(
        Workspace.load(tmp_path),
        models=_TextModels(),  # type: ignore[arg-type]
        metrics=metrics,
    ) as runtime:
        result = await runtime.agent("default").run(secret, timeout_seconds=10)
        assert result.status is ExecutionStatus.SUCCEEDED

    end = datetime.now(timezone.utc) + timedelta(seconds=1)
    window = MetricWindow.between(start, end)
    for metric_name in (
        "linktools.model.request.count",
        "linktools.agent.run.count",
        "linktools.execution.count",
    ):
        query = await metrics.query(MetricQuery(metric_name, window))
        assert len(query.points) == 1
        assert query.points[0].value == 1
        assert query.points[0].sample_count == 1

    observations = []
    for kind in (
        "linktools.model.request",
        "linktools.agent.run",
        "linktools.execution.terminal",
    ):
        observations.extend(
            await store.scan_observations(
                "runtime-metrics",
                kind=kind,
                start=start,
                end=end,
                limit=100,
            )
        )
    assert len(observations) == 3
    assert secret not in repr(observations)


@pytest.mark.asyncio
async def test_runtime_metrics_backend_failure_does_not_change_execution_result(
    tmp_path: Path,
) -> None:
    _write_default_agent(tmp_path)
    metrics = Metrics.from_store(_FailingMetricStore(), namespace="runtime-fail-open")

    async with Runtime.open(
        Workspace.load(tmp_path),
        models=_TextModels(),  # type: ignore[arg-type]
        metrics=metrics,
    ) as runtime:
        result = await runtime.agent("default").run("hello", timeout_seconds=10)
        assert result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_runtime_metric_buffer_is_bounded_and_close_is_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_metrics, "_QUEUE_CAPACITY", 1)
    store = _BlockingMetricStore()
    buffer = runtime_metrics._RuntimeMetricBuffer(
        Metrics.from_store(store, namespace="buffer")
    )

    assert buffer.try_record(_observation("first")) is True
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    assert buffer.try_record(_observation("second")) is True
    assert buffer.try_record(_observation("third")) is False

    store.release.set()
    await buffer.close()
    assert buffer.try_record(_observation("after-close")) is False
    assert sum(len(batch) for batch in store.batches) == 2


@pytest.mark.asyncio
async def test_model_and_tool_producers_do_not_capture_payload_or_exception_text() -> None:
    recorder = _CaptureRecorder()
    capability = _RuntimeMetricCapability(
        recorder,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id="session",
        step_run_id="run",
        agent_id="agent",
        provider="provider",
        model_identity="provider:model",
        route_id="default",
    )
    secret = "DO_NOT_PERSIST_THIS_SECRET"

    async def model_handler(_request: object) -> object:
        raise RuntimeError(f"provider failed with {secret}")

    with pytest.raises(RuntimeError):
        await capability.wrap_model_request(  # type: ignore[arg-type]
            None,
            request_context=object(),
            handler=model_handler,  # type: ignore[arg-type]
        )

    call = ToolCallPart(
        "dangerous_tool",
        {"token": secret},
        tool_call_id="call-1",
    )

    async def tool_handler(_args: dict[str, Any]) -> object:
        raise RuntimeError(f"tool failed with {secret}")

    with pytest.raises(RuntimeError):
        await capability.wrap_tool_execute(  # type: ignore[arg-type]
            None,
            call=call,
            tool_def=ToolDefinition(name="dangerous_tool"),
            args={"token": secret},
            handler=tool_handler,
        )

    assert [item.kind for item in recorder.observations] == [
        "linktools.model.request",
        "linktools.tool.execution",
    ]
    assert recorder.observations[0].error_code == ErrorCode.MODEL_API_ERROR.value
    assert (
        recorder.observations[1].error_code
        == ErrorCode.TOOL_EXECUTION_FAILED.value
    )
    assert secret not in repr(recorder.observations)


@pytest.mark.asyncio
async def test_task_attempt_observation_identity_is_stable_for_same_fence() -> None:
    class Delegate:
        async def complete(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return object()

    recorder = _CaptureRecorder()
    repository = runtime_metrics._MetricTaskRepository(
        Delegate(),  # type: ignore[arg-type]
        recorder,
        source_namespace="workspace",
    )
    lease = TaskLease(
        graph_id="graph",
        node_id="node",
        tenant_id="tenant",
        owner="worker",
        fence=7,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )

    for _ in range(2):
        await repository.complete(
            lease,
            tenant_id="tenant",
            execution_id="execution",
            result_digest="a" * 64,
        )

    assert len(recorder.observations) == 2
    first, second = recorder.observations
    assert first.observation_id == second.observation_id
    assert first.correlation == {
        "graph_id": "graph",
        "node_id": "node",
        "fence": 7,
        "execution_id": "execution",
    }
