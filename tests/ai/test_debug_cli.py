#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local AI debugging command coverage."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    Page,
    Principal,
    UsageMetrics,
)
from linktools.ai.observe import MetricMeasurement, Metrics, Observation
from linktools.ai.runtime import (
    ExecutionHistoryItem,
    ExecutionTraceItem,
    ExecutionView,
    ListExecutionRequest,
    ModelInteractionItem,
    TranscriptItem,
)
from linktools.ai.workspace import Workspace
from linktools.cli import CommandParser
from linktools.commands._ai_common import _local_metrics, _local_runtime_state
from linktools.commands.ai.run import Command as RunCommand
from linktools.commands.ai_history import (
    Command as HistoryCommand,
    _execution_detail,
    _recent_executions,
)
from linktools.commands.ai_metrics import Command as MetricsCommand, _query_summary


def _execution(index: int) -> ExecutionView:
    execution_id = f"exec-{index:03d}"
    return ExecutionView(
        execution_id,
        "default",
        ExecutionStatus.SUCCEEDED,
        ExecutionLineageKind.RUN,
        None,
        execution_id,
        None,
        "session",
    )


class _ListHistory:
    def __init__(self) -> None:
        self.values = tuple(_execution(index) for index in range(25))

    async def list_executions(
        self, request: ListExecutionRequest
    ) -> Page[ExecutionView]:
        start = 0 if request.cursor is None else int(request.cursor)
        end = min(len(self.values), start + 15)
        next_cursor = None if end == len(self.values) else str(end)
        return Page(self.values[start:end], next_cursor)


class _DetailHistory:
    async def inspect_execution(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionView:
        del principal
        assert execution_id == "exec-001"
        return _execution(1)

    async def history(
        self, execution_id: str, **_kwargs: object
    ) -> Page[ExecutionHistoryItem]:
        return Page((ExecutionHistoryItem(execution_id, 1, "tool", {"ok": True}),))

    async def trace(
        self, execution_id: str, **_kwargs: object
    ) -> Page[ExecutionTraceItem]:
        return Page((ExecutionTraceItem(execution_id, 1, {"kind": "TOOL"}),))

    async def transcript(
        self, execution_id: str, **_kwargs: object
    ) -> Page[TranscriptItem]:
        return Page((TranscriptItem(execution_id, 1, "hello"),))

    async def model_interactions(
        self,
        execution_id: str,
        **_kwargs: object,
    ) -> Page[ModelInteractionItem]:
        return Page(
            (
                ModelInteractionItem(
                    execution_id=execution_id,
                    segment_sequence=1,
                    depth=0,
                    request_sequence=1,
                    purpose="run",
                    step_index=0,
                    output_retry_index=None,
                    model={"name": "test"},
                    request={"messages": []},
                    response={"text": "ok"},
                    status="SUCCEEDED",
                    error_code=None,
                    duration_ns=1_000,
                    usage=UsageMetrics(input_tokens=3, output_tokens=2),
                ),
            )
        )


def test_debug_commands_are_top_level_short_names() -> None:
    assert HistoryCommand().name == "ai-history"
    assert MetricsCommand().name == "ai-metrics"


def test_ai_run_no_longer_exposes_storage_selection() -> None:
    parser = CommandParser()
    RunCommand().init_arguments(parser)
    assert "storage" not in {action.dest for action in parser._actions}


@pytest.mark.asyncio
async def test_local_debug_storage_uses_separate_runtime_and_metrics_databases(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path, workspace_id="workspace")
    state = _local_runtime_state(workspace)
    metrics = await _local_metrics(workspace)
    runtime_root = workspace.storage_root / "runtime"

    assert all(
        state.plan.route(domain).path == (runtime_root / "runtime.db").resolve()
        for domain in state.plan.durable_domains
    )
    assert metrics.namespace == workspace.workspace_id
    assert (runtime_root / "metrics.db").is_file()
    assert not (runtime_root / "runtime.db").exists()


@pytest.mark.asyncio
async def test_history_list_returns_latest_twenty_in_reverse_order() -> None:
    principal = Principal("cli", "default", "service")
    values = await _recent_executions(_ListHistory(), principal)

    assert [item.execution_id for item in values] == [
        f"exec-{index:03d}" for index in range(24, 4, -1)
    ]


@pytest.mark.asyncio
async def test_history_detail_combines_public_debug_views() -> None:
    principal = Principal("cli", "default", "service")
    payload = await _execution_detail(_DetailHistory(), principal, "exec-001")

    assert payload["execution"].execution_id == "exec-001"
    assert payload["history"][0].item_kind == "tool"
    assert payload["trace"][0].sequence == 1
    assert payload["transcript"][0].text == "hello"
    assert payload["model_interactions"][0].usage.input_tokens == 3


@pytest.mark.asyncio
async def test_metrics_summary_uses_builtin_metrics() -> None:
    metrics = Metrics.in_memory(namespace="workspace")
    occurred_at = datetime.now(timezone.utc)
    await metrics.record_observations(
        (
            Observation(
                version=1,
                observation_id="execution-observation",
                kind="linktools.execution.terminal",
                occurred_at=occurred_at,
                source_namespace="workspace",
                tenant_id="default",
                status="SUCCEEDED",
                error_code=None,
                correlation={},
                dimensions={},
                measurements=(MetricMeasurement("latency_ns", 1, 2_000_000_000),),
            ),
            Observation(
                version=1,
                observation_id="model-observation",
                kind="linktools.model.request",
                occurred_at=occurred_at,
                source_namespace="workspace",
                tenant_id="default",
                status="SUCCEEDED",
                error_code=None,
                correlation={},
                dimensions={},
                measurements=(
                    MetricMeasurement("latency_ns", 1, 500_000_000),
                    MetricMeasurement("input_tokens", 1, 10),
                    MetricMeasurement("output_tokens", 1, 4),
                    MetricMeasurement("cache_read_tokens", 1, 3),
                    MetricMeasurement("cache_write_tokens", 1, 2),
                ),
            ),
            Observation(
                version=1,
                observation_id="tool-observation",
                kind="linktools.tool.execution",
                occurred_at=occurred_at,
                source_namespace="workspace",
                tenant_id="default",
                status="SUCCEEDED",
                error_code=None,
                correlation={},
                dimensions={},
                measurements=(),
            ),
        )
    )

    summary = await _query_summary(metrics)
    values = {
        metric: result.points[0].value
        for _, metric, result in summary
    }

    assert values["linktools.execution.count"] == 1
    assert values["linktools.execution.failure_ratio"] == 0
    assert values["linktools.execution.latency"] == 2_000_000_000
    assert values["linktools.model.request.count"] == 1
    assert values["linktools.model.input_tokens"] == 10
    assert values["linktools.model.output_tokens"] == 4
    assert values["linktools.model.cache_read_tokens"] == 3
    assert values["linktools.model.cache_write_tokens"] == 2
    assert values["linktools.tool.execution.count"] == 1
