#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local AI debugging command coverage."""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    TenantAuthorizationPolicy,
    UsageMetrics,
)
from linktools.ai.errors import AIError, ErrorCode, ErrorDiagnostics
from linktools.ai.observe import MetricMeasurement, Metrics, Observation
from linktools.ai.runtime import (
    ExecutionHistoryItem,
    ExecutionInfo,
    ModelInteractionItem,
    RuntimeHistory,
    TranscriptItem,
)
from linktools.ai.workspace import Workspace
from linktools.commands.ai._common import (
    _load_workspace,
    _local_metrics,
    _local_runtime_state,
)
from linktools.commands.ai.history import Command as HistoryCommand, _emit_execution_detail
from linktools.commands.ai.metrics import Command as MetricsCommand, _query_summary
from linktools.commands.ai.run import Command as RunCommand


def _info(
    execution_id: str,
    created_at: datetime,
    *,
    error_code: str | None = None,
) -> ExecutionInfo:
    return ExecutionInfo(
        execution_id=execution_id,
        agent_id="agent",
        status=(
            ExecutionStatus.FAILED
            if error_code is not None
            else ExecutionStatus.SUCCEEDED
        ),
        lineage_kind=ExecutionLineageKind.RUN,
        parent_execution_id=None,
        root_execution_id=execution_id,
        parent_invocation_id=None,
        session_id="session",
        created_at=created_at,
        updated_at=created_at,
        error_code=error_code,
        safe_error_details={} if error_code is None else {"stage": "runtime"},
        error_diagnostics=(
            None
            if error_code is None
            else ErrorDiagnostics("RuntimeError", "failed", "0" * 64)
        ),
    )


def _record(
    execution_id: str,
    created_at: datetime,
    *,
    error_code: str | None = None,
):
    info = _info(execution_id, created_at, error_code=error_code)
    return SimpleNamespace(
        **{
            field: getattr(info, field)
            for field in (
                "execution_id",
                "agent_id",
                "status",
                "lineage_kind",
                "parent_execution_id",
                "root_execution_id",
                "parent_invocation_id",
                "session_id",
                "created_at",
                "updated_at",
                "error_code",
                "safe_error_details",
                "error_diagnostics",
            )
        }
    )


class _RecentRepository:
    def __init__(self, records: tuple[object, ...]) -> None:
        self.records = records
        self.calls = 0

    async def list_candidates(self, **kwargs):
        self.calls += 1
        cursor = kwargs["cursor"]
        start = 0 if cursor is None else int(cursor) + 1
        end = min(len(self.records), start + 2)
        items = tuple(
            SimpleNamespace(record=self.records[index], cursor=str(index))
            for index in range(start, end)
        )
        return SimpleNamespace(items=items, has_more=end < len(self.records))

    async def get_header(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if any(value.execution_id == execution_id for value in self.records):
            return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)
        return None

    async def get(self, execution_id: str, *, tenant_id: str):
        del tenant_id
        return next(
            (
                value
                for value in self.records
                if value.execution_id == execution_id
            ),
            None,
        )


class _DetailHistory:
    def __init__(self, execution: ExecutionInfo) -> None:
        self.execution = execution
        self.trace_called = False

    async def inspect_execution(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionInfo:
        del principal
        assert execution_id == self.execution.execution_id
        return self.execution

    async def history(self, execution_id: str, **kwargs) -> Page[ExecutionHistoryItem]:
        cursor = kwargs["cursor"]
        if cursor is None:
            return Page(
                (ExecutionHistoryItem(execution_id, 1, "tool", {"page": 1}),),
                "next",
            )
        return Page((ExecutionHistoryItem(execution_id, 2, "tool", {"page": 2}),))

    async def transcript(self, execution_id: str, **_kwargs) -> Page[TranscriptItem]:
        return Page((TranscriptItem(execution_id, 1, "hello"),))

    async def model_interactions(
        self,
        execution_id: str,
        **_kwargs,
    ) -> Page[ModelInteractionItem]:
        return Page(
            (
                ModelInteractionItem(
                    execution_id=execution_id,
                    segment_sequence=1,
                    depth=0,
                    request_sequence=1,
                    purpose="agent",
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

    async def trace(self, *_args, **_kwargs):
        self.trace_called = True
        raise AssertionError("ai history must not materialize trace by default")


def test_debug_commands_use_ai_group_and_minimal_names() -> None:
    assert HistoryCommand().name == "history"
    assert MetricsCommand().name == "metrics"
    history_actions = {
        action.dest for action in HistoryCommand().create_parser()._actions
    }
    metrics_actions = {
        action.dest for action in MetricsCommand().create_parser()._actions
    }
    assert history_actions == {"help", "execution_id"}
    assert metrics_actions == {"help", "metric"}


def test_ai_run_no_longer_exposes_storage_selection() -> None:
    parser = RunCommand().create_parser()
    assert "storage" not in {action.dest for action in parser._actions}


def test_workspace_discovery_walks_up_from_nested_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path, workspace_id="workspace")
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    loaded = _load_workspace()

    assert loaded.root == workspace.root
    assert not (nested / ".linktools").exists()


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
async def test_existing_invalid_metrics_database_fails_before_runtime_use(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path, workspace_id="workspace")
    runtime_root = workspace.storage_root / "runtime"
    runtime_root.mkdir(parents=True)
    database = runtime_root / "metrics.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE marker (id INTEGER PRIMARY KEY)")

    with pytest.raises(AIError) as error:
        await _local_metrics(workspace)

    assert error.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING


@pytest.mark.asyncio
async def test_runtime_history_returns_exact_recent_executions_and_errors() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = (
        _record("exec-middle", base + timedelta(minutes=2)),
        _record("exec-old", base),
        _record("exec-new", base + timedelta(minutes=3), error_code="FAILED_CODE"),
        _record("exec-newer", base + timedelta(minutes=4)),
        _record("exec-less-old", base + timedelta(minutes=1)),
    )
    repository = _RecentRepository(records)
    history = RuntimeHistory(
        SimpleNamespace(),
        tenant_id="default",
        executions=repository,
        authorization=TenantAuthorizationPolicy("default"),
    )
    principal = Principal("cli", "default", "service")

    recent = await history.recent_executions(principal=principal, limit=3)
    inspected = await history.inspect_execution("exec-new", principal=principal)

    assert [value.execution_id for value in recent] == [
        "exec-newer",
        "exec-new",
        "exec-middle",
    ]
    assert repository.calls == 3
    assert inspected.error_code == "FAILED_CODE"
    assert inspected.safe_error_details == {"stage": "runtime"}
    assert inspected.error_diagnostics is not None


@pytest.mark.asyncio
async def test_history_detail_streams_pages_without_trace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    execution = _info(
        "exec-001",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        error_code="FAILED_CODE",
    )
    history = _DetailHistory(execution)
    principal = Principal("cli", "default", "service")

    await _emit_execution_detail(history, principal, "exec-001")

    output = capsys.readouterr().out
    assert '"error_code": "FAILED_CODE"' in output
    assert '"page": 1' in output
    assert '"page": 2' in output
    assert "hello" in output
    assert "Model interactions" in output
    assert "Trace" not in output
    assert history.trace_called is False


@pytest.mark.asyncio
async def test_metrics_summary_uses_one_window_and_builtin_metrics() -> None:
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
                measurements=(
                    MetricMeasurement("latency_ns", 1, 2_000_000_000),
                ),
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
    values = {metric: result.points[0].value for _, metric, result in summary}
    windows = {
        (result.window_start, result.window_end)
        for _, _, result in summary
    }

    assert len(windows) == 1
    assert values["linktools.execution.count"] == 1
    assert values["linktools.execution.failure_ratio"] == 0
    assert values["linktools.execution.latency"] == 2_000_000_000
    assert values["linktools.model.request.count"] == 1
    assert values["linktools.model.input_tokens"] == 10
    assert values["linktools.model.output_tokens"] == 4
    assert values["linktools.model.cache_read_tokens"] == 3
    assert values["linktools.model.cache_write_tokens"] == 2
    assert values["linktools.tool.execution.count"] == 1
