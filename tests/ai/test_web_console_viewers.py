#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP inspection parity for persisted executions, sessions and metrics."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import httpx
import pytest

pytest.importorskip("starlette")

from linktools.ai.core import (
    ExecutionLineageKind, ExecutionStatus, Page, Principal, SessionStatus,
    UsageMetrics,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import MetricMeasurement, Metrics, Observation
from linktools.ai.runtime import (
    ExecutionHistoryItem, ExecutionInfo, ExecutionTraceItem, ListExecutionRequest,
    ModelInteractionItem, SessionView, TranscriptItem, UsageSummary,
)
from linktools.ai.web import create_app


def _client(app: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765",
    )


def _execution() -> ExecutionInfo:
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ExecutionInfo(
        execution_id="execution",
        binding_kind="agent",
        agent_id="agent",
        task_id=None,
        status=ExecutionStatus.FAILED,
        lineage_kind=ExecutionLineageKind.RUN,
        parent_execution_id=None,
        root_execution_id="execution",
        parent_invocation_id=None,
        session_id="session",
        created_at=created,
        updated_at=created,
        started_at=created,
        terminal_at=created,
        binding_digest="a" * 64,
        usage=UsageSummary(),
        error_code="FAILED_CODE",
        safe_error_details={"stage": "runtime"},
        error_diagnostics=None,
    )


def _model(request_seq: int) -> ModelInteractionItem:
    return ModelInteractionItem(
        execution_id="execution",
        agent_run_seq=1,
        depth=0,
        model_request_seq=request_seq,
        purpose="agent",
        step_index=request_seq - 1,
        output_retry_index=None,
        model={"name": "test"},
        request={
            "instructions": ["duplicate instruction mirror"],
            "messages": [{"kind": "request", "parts": [
                {"part_kind": "system-prompt", "content": "real system prompt"},
                {"part_kind": "user-prompt", "content": [
                    "hello",
                    {"media_type": "image/png", "size": 2048, "digest": "a" * 64},
                ]},
            ]}],
            "parameters": {
                "instruction_parts": [
                    {"content": "fixed workspace guidance", "name": "workspace", "dynamic": False},
                    {"content": "repository overlay", "name": "repository", "dynamic": True},
                ],
                "function_tools": [{"name": "read_file"}],
                "native_tools": [],
                "revealed_tool_names": ["read_file"],
                "deferred_capability_ids": [],
                "output_mode": "text",
                "allow_text_output": True,
                "allow_image_output": False,
            },
        },
        response={"text": f"response {request_seq}"},
        status="SUCCEEDED",
        error_code=None,
        duration_ns=1_000,
        usage=UsageMetrics(
            input_tokens=3, output_tokens=2, cache_read_tokens=1, cache_write_tokens=4,
        ),
    )


class _ExecutionListHistory:
    tenant_id = "default"

    def __init__(self) -> None:
        self.allow_recent_scan = False
        oldest = _execution()
        self.records = (
            replace(oldest, execution_id="newest", created_at=oldest.created_at + timedelta(minutes=2)),
            replace(oldest, execution_id="middle", created_at=oldest.created_at + timedelta(minutes=1)),
            replace(oldest, execution_id="oldest"),
        )

    async def list_executions(
        self, request: ListExecutionRequest,
    ) -> Page[ExecutionInfo]:
        assert request.principal.tenant_id == self.tenant_id and request.limit == 20
        return Page(tuple(reversed(self.records)))

    async def recent_executions(
        self, *, principal: Principal, limit: int,
    ) -> tuple[ExecutionInfo, ...]:
        assert self.allow_recent_scan, "Ordinary browsing must not scan all execution metadata"
        assert principal.tenant_id == self.tenant_id and limit == 20
        return self.records


@pytest.mark.asyncio
async def test_execution_list_scans_for_newest_only_when_explicitly_requested() -> None:
    history = _ExecutionListHistory()
    async with _client(create_app(history=history)) as http:  # type: ignore[arg-type]
        ordinary = await http.get("/api/executions?limit=20")
        assert ordinary.status_code == 200, ordinary.text
        assert [item["execution_id"] for item in ordinary.json()["items"]] == [
            "oldest", "middle", "newest",
        ]
        history.allow_recent_scan = True
        recent = await http.get("/api/executions?recent=true&limit=20")
        assert recent.status_code == 200, recent.text
        page = recent.json()
        assert page["recent_scan"] is True and page["next_cursor"] is None
        assert [item["execution_id"] for item in page["items"]] == [
            "newest", "middle", "oldest",
        ]
        history.allow_recent_scan = False
        for key in ("cursor", "session_id", "agent_id", "parent_execution_id"):
            rejected = await http.get(
                "/api/executions", params={"recent": "true", "limit": 20, key: "value"},
            )
            assert rejected.status_code == 400, (key, rejected.text)


class _PagedHistory:
    tenant_id = "default"

    async def inspect_execution(
        self, execution_id: str, *, principal: Principal,
    ) -> ExecutionInfo:
        assert execution_id == "execution"
        assert principal.tenant_id == self.tenant_id
        return _execution()

    async def history(
        self, execution_id: str, *, principal: Principal, cursor: str | None,
        include_content: bool, limit: int,
    ) -> Page[ExecutionHistoryItem]:
        assert execution_id == "execution" and include_content and limit == 1
        assert principal.tenant_id == self.tenant_id
        assert cursor in {None, "history-next"}
        sequence = 1 if cursor is None else 2
        return Page(
            (ExecutionHistoryItem(execution_id, sequence, "tool", {"page": sequence}),),
            "history-next" if cursor is None else None,
        )

    async def transcript(
        self, execution_id: str, *, principal: Principal, cursor: str | None,
        include_content: bool, limit: int,
    ) -> Page[TranscriptItem]:
        assert execution_id == "execution" and include_content and limit == 1
        assert principal.tenant_id == self.tenant_id
        assert cursor in {None, "transcript-next"}
        sequence = 1 if cursor is None else 2
        return Page(
            (TranscriptItem(execution_id, sequence, "hello" if cursor is None else "world"),),
            "transcript-next" if cursor is None else None,
        )

    async def model_interactions(
        self, execution_id: str, *, principal: Principal, cursor: str | None,
        include_content: bool, limit: int,
    ) -> Page[ModelInteractionItem]:
        assert execution_id == "execution" and include_content and limit == 1
        assert principal.tenant_id == self.tenant_id
        assert cursor in {None, "models-next"}
        return Page(
            (_model(1 if cursor is None else 2),),
            "models-next" if cursor is None else None,
        )

    async def trace(
        self, execution_id: str, **kwargs: object,
    ) -> Page[ExecutionTraceItem]:
        raise AssertionError("Unselected trace content must not be materialized")


@pytest.mark.asyncio
async def test_execution_viewers_page_history_transcript_and_models_without_eager_trace() -> None:
    async with _client(create_app(history=_PagedHistory())) as http:  # type: ignore[arg-type]
        response = await http.get("/api/executions/execution")
        assert response.status_code == 200, response.text
        execution = response.json()
        assert execution["error_code"] == "FAILED_CODE"
        assert execution["safe_error_details"] == {"stage": "runtime"}
        assert execution["error_diagnostics"] is None
        pages = {}
        for kind in ("history", "transcript", "models"):
            items = []
            cursor = None
            while True:
                response = await http.get(
                    f"/api/executions/execution/{kind}",
                    params={"limit": 1, "include_content": "true", **({"cursor": cursor} if cursor else {})},
                )
                assert response.status_code == 200, response.text
                page = response.json()
                items.extend(page["items"])
                cursor = page["next_cursor"]
                if cursor is None:
                    break
            pages[kind] = items

    assert [item["content"] for item in pages["history"]] == [{"page": 1}, {"page": 2}]
    assert [item["text"] for item in pages["transcript"]] == ["hello", "world"]
    assert [item["model_request_seq"] for item in pages["models"]] == [1, 2]
    assert [item["response"]["text"] for item in pages["models"]] == ["response 1", "response 2"]
    model = pages["models"][0]
    assert model["model"] == {"name": "test"}
    assert model["duration_ns"] == 1_000
    assert model["usage"]["cache_read_tokens"] == 1
    assert model["usage"]["cache_write_tokens"] == 4
    assert model["request"]["parameters"]["instruction_parts"] == [
        {"content": "fixed workspace guidance", "name": "workspace", "dynamic": False},
        {"content": "repository overlay", "name": "repository", "dynamic": True},
    ]
    assert model["request"]["messages"][0]["parts"][1]["content"][1] == {
        "media_type": "image/png", "size": 2048, "digest": "a" * 64,
    }


class _TraceHistory(_PagedHistory):
    async def trace(
        self, execution_id: str, *, principal: Principal, cursor: str | None,
        limit: int,
    ) -> Page[ExecutionTraceItem]:
        assert execution_id == "execution" and cursor is None
        assert principal.tenant_id == self.tenant_id
        return Page((ExecutionTraceItem(execution_id, 1, {
            "kind": "MODEL_RESPONSE",
            "status": "SUCCEEDED",
            "step_index": 0,
            "agent_run_seq": 1,
            "scope": "root",
            "model_request_seq": 1,
            "purpose": "agent",
            "duration_ns": 2_000_000,
            "token_usage": {
                "input_tokens": 10, "output_tokens": 4,
                "cache_read_tokens": 0, "cache_write_tokens": 0,
            },
        }),))


@pytest.mark.asyncio
async def test_trace_viewer_preserves_compact_model_timing_and_usage() -> None:
    async with _client(create_app(history=_TraceHistory())) as http:  # type: ignore[arg-type]
        response = await http.get("/api/executions/execution/trace")
        assert response.status_code == 200, response.text
        page = response.json()

    assert page["next_cursor"] is None
    assert len(page["items"]) == 1
    item = page["items"][0]
    assert item["execution_id"] == "execution"
    assert item["step_event_seq"] == 1
    assert item["payload"] == {
        "kind": "MODEL_RESPONSE", "status": "SUCCEEDED", "step_index": 0,
        "agent_run_seq": 1, "scope": "root", "model_request_seq": 1,
        "purpose": "agent", "duration_ns": 2_000_000,
        "token_usage": {
            "input_tokens": 10, "output_tokens": 4,
            "cache_read_tokens": 0, "cache_write_tokens": 0,
        },
    }


class _SessionHistory:
    tenant_id = "default"

    async def recent_sessions(
        self, *, principal: Principal, limit: int,
    ) -> tuple[SessionView, ...]:
        assert principal.tenant_id == self.tenant_id and limit == 20
        return (SessionView(
            "session", "agent", SessionStatus.OPEN, revision=2, cwd="src",
            active_execution_id="execution", metadata={"title": "Current work"},
            history_quality="complete",
        ),)


@pytest.mark.asyncio
async def test_session_viewer_preserves_recent_session_metadata() -> None:
    async with _client(create_app(history=_SessionHistory())) as http:  # type: ignore[arg-type]
        response = await http.get("/api/sessions?limit=20")
        assert response.status_code == 200, response.text
        page = response.json()

    assert page["recent_only"] is True
    assert page["next_cursor"] is None
    assert page["items"] == [{
        "session_id": "session", "agent_id": "agent", "status": "OPEN",
        "revision": 2, "cwd": "src", "active_execution_id": "execution",
        "metadata": {"title": "Current work"}, "history_quality": "complete",
    }]


class _UnavailableSessionTimeline:
    tenant_id = "default"

    def __init__(self) -> None:
        self.allow_timeline = False

    async def inspect_session(
        self, session_id: str, *, principal: Principal,
    ) -> SessionView:
        assert session_id == "session" and principal.tenant_id == self.tenant_id
        return SessionView(
            session_id, "agent", SessionStatus.OPEN, revision=3, cwd="src",
            active_execution_id="execution", metadata={"title": "Readable metadata"},
            history_quality="complete",
        )

    async def session_timeline(
        self, session_id: str, *, principal: Principal, cursor: str | None,
        limit: int,
    ) -> Page[object]:
        assert self.allow_timeline, "Metadata-only inspection must not load timeline content"
        assert session_id == "session" and principal.tenant_id == self.tenant_id
        raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)


@pytest.mark.asyncio
async def test_session_metadata_is_readable_without_loading_unavailable_timeline() -> None:
    history = _UnavailableSessionTimeline()
    async with _client(create_app(history=history)) as http:  # type: ignore[arg-type]
        metadata = await http.get(
            "/api/session", params={"session_id": "session", "include_timeline": "false"},
        )
        assert metadata.status_code == 200, metadata.text
        assert metadata.json() == {
            "session": {
                "session_id": "session", "agent_id": "agent", "status": "OPEN",
                "revision": 3, "cwd": "src", "active_execution_id": "execution",
                "metadata": {"title": "Readable metadata"}, "history_quality": "complete",
            },
            "timeline": None,
        }

        history.allow_timeline = True
        bundled = await http.get("/api/session", params={"session_id": "session"})
        assert bundled.status_code == 503, bundled.text
        assert bundled.json()["code"] == ErrorCode.SESSION_HISTORY_UNAVAILABLE.value
        assert "timeline" not in bundled.json()


@pytest.mark.asyncio
async def test_metrics_summary_uses_one_window_and_all_builtin_metrics() -> None:
    metrics = Metrics.in_memory(namespace="workspace")
    occurred_at = datetime.now(timezone.utc)
    await metrics.record_observations(tuple(
        Observation(
            version=1, observation_id=identity, kind=kind, occurred_at=occurred_at,
            source_namespace="workspace", tenant_id="default", status="SUCCEEDED",
            error_code=None, correlation={}, dimensions={}, measurements=measurements,
        )
        for identity, kind, measurements in (
            ("execution-observation", "linktools.execution.terminal", (
                MetricMeasurement("latency_ns", 1, 2_000_000_000),
            )),
            ("model-observation", "linktools.model.request", (
                MetricMeasurement("latency_ns", 1, 500_000_000),
                MetricMeasurement("input_tokens", 1, 10),
                MetricMeasurement("output_tokens", 1, 4),
                MetricMeasurement("cache_read_tokens", 1, 3),
                MetricMeasurement("cache_write_tokens", 1, 2),
            )),
            ("tool-observation", "linktools.tool.execution", ()),
        )
    ))

    async with _client(create_app(metrics=metrics)) as http:
        response = await http.get("/api/metrics")
        assert response.status_code == 200, response.text
        results = response.json()["items"]
        named = await http.get("/api/metrics?metric=linktools.model.input_tokens")
        assert named.status_code == 200, named.text

    assert len(results) == 12
    windows = {(result["window_start"], result["window_end"]) for result in results}
    assert len(windows) == 1
    start, end = next(iter(windows))
    assert datetime.fromisoformat(end) - datetime.fromisoformat(start) == timedelta(days=1)
    assert datetime.fromisoformat(start) <= occurred_at <= datetime.fromisoformat(end)
    assert all(len(result["points"]) == 1 for result in results)
    assert {result["metric"]: result["points"][0]["value"] for result in results} == {
        "linktools.execution.count": 1,
        "linktools.execution.failure_ratio": 0,
        "linktools.execution.latency": 2_000_000_000,
        "linktools.model.request.count": 1,
        "linktools.model.request.failure_ratio": 0,
        "linktools.model.request.latency": 500_000_000,
        "linktools.tool.execution.count": 1,
        "linktools.tool.execution.failure_ratio": 0,
        "linktools.model.input_tokens": 10,
        "linktools.model.output_tokens": 4,
        "linktools.model.cache_read_tokens": 3,
        "linktools.model.cache_write_tokens": 2,
    }
    named_results = named.json()["items"]
    assert len(named_results) == 1
    assert named_results[0]["metric"] == "linktools.model.input_tokens"
    assert named_results[0]["points"][0]["value"] == 10


@pytest.mark.asyncio
async def test_metrics_summary_preserves_empty_values_and_cache_metrics() -> None:
    async with _client(create_app(metrics=Metrics.in_memory(namespace="workspace"))) as http:
        response = await http.get("/api/metrics")
        assert response.status_code == 200, response.text
        results = {item["metric"]: item for item in response.json()["items"]}

    assert len(results) == 12
    assert all(len(item["points"]) == 1 for item in results.values())
    assert all(item["points"][0]["sample_count"] == 0 for item in results.values())
    assert results["linktools.execution.count"]["points"][0]["value"] == 0
    assert results["linktools.execution.failure_ratio"]["points"][0]["value"] is None
    assert "linktools.model.cache_read_tokens" in results
    assert "linktools.model.cache_write_tokens" in results
