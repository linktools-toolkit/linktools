#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public read-only Runtime history composition coverage."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    Principal,
    PrincipalKind,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    UsageMetrics,
    TenantAuthorizationPolicy,
    canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode, ErrorDiagnostics
from linktools.ai.runtime import (
    ArtifactView,
    Execution,
    ExecutionEvent,
    ExecutionHistoryItem,
    ExecutionTraceItem,
    Page,
    TranscriptItem,
    RuntimeExecutions,
    RuntimeStorage,
    TaskGraphRun,
    UsageSummary,
)
from linktools.ai.runtime._history_service import DefaultExecutionHistoryService
from linktools.ai.runtime._runtime_history import RuntimeHistory
from linktools.ai.runtime.state._contracts import StoredUserInput
from linktools.ai.storage import StoredPayload
from linktools.ai.task import (
    TaskBindingContract,
    TaskGraph,
    TaskGraphState,
    TaskNode,
    TaskNodeView,
    TaskResultRecord,
    TaskStatus,
)


class _Executions:
    tenant_id = "tenant"

    async def get_header(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)
        return None

    async def get(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> object | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return SimpleNamespace(execution_id=execution_id, tenant_id=tenant_id)
        return None


class _Reader:
    async def history(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
        message_seq: int | None = None,
        part_index: int | None = None,
    ) -> Page[ExecutionHistoryItem]:
        assert tenant_id == "tenant"
        assert cursor is None
        assert limit == 100
        self.history_filters = {
            "agent_run_seq": agent_run_seq,
            "model_request_seq": model_request_seq,
            "step_index": step_index,
            "tool_call_id": tool_call_id,
            "message_seq": message_seq,
            "part_index": part_index,
        }
        return Page((ExecutionHistoryItem(
            execution_id,
            0 if message_seq is None else message_seq,
            "user",
            "hello",
            agent_run_seq=agent_run_seq,
            model_request_seq=model_request_seq,
            step_index=step_index,
            tool_call_id=tool_call_id,
            part_index=part_index,
        ),))

    async def trace(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> Page[ExecutionTraceItem]:
        assert tenant_id == "tenant"
        self.trace_filters = {
            "agent_run_seq": agent_run_seq,
            "model_request_seq": model_request_seq,
            "step_index": step_index,
            "tool_call_id": tool_call_id,
        }
        return Page((ExecutionTraceItem(execution_id, 0, {"kind": "TEST"}),))

    async def transcript(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> Page[TranscriptItem]:
        assert tenant_id == "tenant"
        return Page((TranscriptItem(execution_id, 0, "hello"),))


@pytest.mark.asyncio
async def test_execution_history_service_owns_authorization_boundary() -> None:
    service = DefaultExecutionHistoryService(
        _Executions(),
        TenantAuthorizationPolicy("tenant"),
        _Reader(),
    )
    principal = Principal("caller", "tenant", "service")

    history = await service.history("execution", principal=principal)
    trace = await service.trace("execution", principal=principal)
    transcript = await service.transcript("execution", principal=principal)

    assert history.items[0].content is None
    assert history.items[0].content_included is False
    assert trace.items[0].payload == {"kind": "TEST"}
    assert transcript.items[0].text is None
    assert transcript.items[0].content_included is False

    raw_history = await service.history(
        "execution",
        principal=principal,
        include_content=True,
    )
    raw_transcript = await service.transcript(
        "execution",
        principal=principal,
        include_content=True,
    )
    assert raw_history.items[0].content == "hello"
    assert raw_history.items[0].content_included is True
    assert raw_transcript.items[0].text == "hello"
    assert raw_transcript.items[0].content_included is True

    with pytest.raises(AIError) as error:
        await service.history(
            "execution",
            principal=Principal("caller", "other-tenant", "service"),
        )
    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


class _PagingReader(_Reader):
    async def history(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
        message_seq: int | None = None,
        part_index: int | None = None,
    ) -> Page[ExecutionHistoryItem]:
        assert tenant_id == "tenant"
        if cursor is None:
            return Page(
                (ExecutionHistoryItem(execution_id, 0, "user", "hello"),),
                "inner-next",
            )
        assert cursor == "inner-next"
        return Page(
            (ExecutionHistoryItem(execution_id, 1, "assistant", "world"),),
            None,
        )

    async def trace(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> Page[ExecutionTraceItem]:
        assert tenant_id == "tenant"
        assert cursor in (None, "inner-next")
        return Page(
            (ExecutionTraceItem(execution_id, 0 if cursor is None else 1, {"kind": "TEST"}),),
            "inner-next" if cursor is None else None,
        )


@pytest.mark.asyncio
async def test_history_cursor_binds_content_mode() -> None:
    service = DefaultExecutionHistoryService(
        _Executions(),
        TenantAuthorizationPolicy("tenant"),
        _PagingReader(),
        HmacCursorSigner("public-history", b"public-history-key"),
    )
    principal = Principal("caller", "tenant", "service")

    page = await service.history("execution", principal=principal)
    assert page.next_cursor is not None
    with pytest.raises(AIError) as raised:
        await service.history(
            "execution",
            principal=principal,
            cursor=page.next_cursor,
            include_content=True,
        )
    assert raised.value.code is ErrorCode.CURSOR_INVALID

    second = await service.history(
        "execution",
        principal=principal,
        cursor=page.next_cursor,
    )
    assert second.items[0].content is None
    assert second.items[0].content_included is False


class _EventExecutions:
    async def get_header(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)
        return None

    async def get(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> object | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return SimpleNamespace(
                execution_id=execution_id,
                tenant_id=tenant_id,
                event_seq=2,
            )
        return None


class _Events:
    def __init__(self) -> None:
        self.values = (
            ExecutionEvent("execution", 1, "STARTED", {"secret": "one"}),
            ExecutionEvent("execution", 2, "SUCCEEDED", {"secret": "two"}),
            ExecutionEvent("execution", 3, "LATE", {"secret": "late"}),
        )

    async def list(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        after_event_seq: int,
        limit: int,
    ) -> Page[ExecutionEvent]:
        assert execution_id == "execution"
        assert tenant_id == "tenant"
        selected = tuple(
            value
            for value in self.values
            if value.event_seq > after_event_seq
        )[:limit]
        return Page(selected, None)


@pytest.mark.asyncio
async def test_runtime_history_execution_events_use_fixed_safe_cutoff() -> None:
    principal = Principal("caller", "tenant", "service")
    history = RuntimeHistory(
        DefaultExecutionHistoryService(
            _EventExecutions(),  # type: ignore[arg-type]
            TenantAuthorizationPolicy("tenant"),
            _Reader(),
        ),
        tenant_id="tenant",
        executions=_EventExecutions(),  # type: ignore[arg-type]
        events=_Events(),  # type: ignore[arg-type]
        authorization=TenantAuthorizationPolicy("tenant"),
        cursor_signer=HmacCursorSigner(
            "runtime-history",
            b"runtime-history-key",
        ),
    )

    first = await history.list_execution_events(
        "execution",
        principal=principal,
        limit=1,
    )
    assert [event.event_seq for event in first.items] == [1]
    assert first.items[0].payload == {}
    assert first.next_cursor is not None

    with pytest.raises(AIError) as raised:
        await history.list_execution_events(
            "execution",
            principal=principal,
            cursor=first.next_cursor,
            include_content=True,
            limit=1,
        )
    assert raised.value.code is ErrorCode.CURSOR_INVALID

    second = await history.list_execution_events(
        "execution",
        principal=principal,
        cursor=first.next_cursor,
        limit=1,
    )
    assert [event.event_seq for event in second.items] == [2]
    assert second.items[0].payload == {}
    assert second.next_cursor is None


@pytest.mark.asyncio
async def test_runtime_history_opens_without_model_or_agent_composition() -> None:
    async with RuntimeHistory.open(
        "workspace",
        storage=RuntimeStorage.in_memory(),
    ) as history:
        assert history.tenant_id == "default"
        with pytest.raises(AIError) as error:
            await history.history(
                "missing-execution",
                principal=Principal("caller", "default", "service"),
            )
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


class _Sessions:
    def __init__(self) -> None:
        from datetime import datetime, timedelta, timezone

        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.records = (
            SimpleNamespace(
                session_id="older",
                agent_id="default",
                status=SessionStatus.OPEN,
                revision=1,
                cwd=".",
                active_execution_id=None,
                history_quality="complete",
                metadata={},
                created_at=base,
                updated_at=base,
                tenant_id="tenant",
                owner_principal_id="runtime",
            ),
            SimpleNamespace(
                session_id="newer",
                agent_id="auditor",
                status=SessionStatus.OPEN,
                revision=3,
                cwd=None,
                active_execution_id="execution",
                history_quality="complete",
                metadata={},
                created_at=base + timedelta(minutes=1),
                updated_at=base + timedelta(minutes=2),
                tenant_id="tenant",
                owner_principal_id="runtime",
            ),
        )

    async def list_page(
        self,
        *,
        tenant_id: str,
        owner_principal_id: str | None,
        cursor: str | None,
        limit: int,
        snapshot: int | None = None,
    ) -> tuple[int, Page[object]]:
        del limit
        values = tuple(
            record
            for record in self.records
            if record.tenant_id == tenant_id
            and (
                owner_principal_id is None
                or record.owner_principal_id == owner_principal_id
            )
        )
        start = 0 if cursor is None else int(cursor)
        end = min(len(values), start + 1)
        next_cursor = None if end == len(values) else str(end)
        return (
            1 if snapshot is None else snapshot,
            Page(values[start:end], next_cursor),
        )

    async def get_header(
        self,
        session_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        record = next(
            (
                value
                for value in self.records
                if value.session_id == session_id and value.tenant_id == tenant_id
            ),
            None,
        )
        if record is None:
            return None
        return ResourceRef(
            ResourceKind.SESSION,
            record.session_id,
            record.tenant_id,
            record.owner_principal_id,
        )

    async def get(
        self,
        session_id: str,
        *,
        tenant_id: str,
    ) -> object | None:
        return next(
            (
                value
                for value in self.records
                if value.session_id == session_id and value.tenant_id == tenant_id
            ),
            None,
        )


@pytest.mark.asyncio
async def test_runtime_history_projects_owned_sessions_without_runtime_open() -> None:
    sessions = _Sessions()
    history = RuntimeHistory(
        SimpleNamespace(),
        tenant_id="tenant",
        sessions=sessions,  # type: ignore[arg-type]
        authorization=TenantAuthorizationPolicy("tenant"),
    )
    principal = Principal(
        "runtime",
        "tenant",
        PrincipalKind.LOCAL_TRUSTED.value,
    )

    recent = await history.recent_sessions(principal=principal, limit=20)
    selected = await history.inspect_session("newer", principal=principal)

    assert [item.session_id for item in recent] == ["newer", "older"]
    assert selected.agent_id == "auditor"
    assert selected.active_execution_id == "execution"

    with pytest.raises(AIError) as denied:
        await history.inspect_session(
            "newer",
            principal=Principal("other", "tenant", PrincipalKind.LOCAL_TRUSTED.value),
        )
    assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED


class _ResultExecutions:
    def __init__(self) -> None:
        self.binding = TaskBindingContract(
            "handler",
            1,
            "none",
            {},
            None,
            1,
            0,
        )
        created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        started_at = created_at + timedelta(seconds=1)
        terminal_at = created_at + timedelta(seconds=2)
        stored_input = StoredUserInput(
            "task-input-v1",
            StoredPayload.inline_json({"value": 1}),
            {"version": 1, "kind": "task"},
        )
        output = StoredPayload.inline_json(None)
        self.result = SimpleNamespace(
            output=output,
            usage=UsageMetrics(),
            created_at=terminal_at,
        )
        self.record = SimpleNamespace(
            execution_id="execution",
            tenant_id="tenant",
            binding_kind="task",
            agent_id=None,
            task_id="handler",
            status=ExecutionStatus.SUCCEEDED,
            lineage_kind=ExecutionLineageKind.RUN,
            parent_execution_id=None,
            root_execution_id="execution",
            parent_invocation_id=None,
            session_id=None,
            created_at=created_at,
            updated_at=terminal_at + timedelta(seconds=10),
            started_at=started_at,
            binding=self.binding,
            binding_digest=self.binding.binding_digest,
            stored_user_input=stored_input,
            result=self.result,
            error_code=None,
            safe_error_details={},
            error_diagnostics=None,
        )

    async def get_header(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)
        return None

    async def get(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> object | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return self.record
        return None

    async def get_result(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> object | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return self.result
        return None


class _InspectionService:
    async def usage(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> UsageSummary:
        assert execution_id == "execution"
        assert principal.tenant_id == "tenant"
        return UsageSummary(
            logical_requests=1,
            succeeded_requests=1,
            input_tokens=2,
            output_tokens=3,
            model_duration_ns=4,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (ExecutionStatus.FAILED, ExecutionStatus.CANCELLED))
async def test_history_result_rejects_invalid_terminal_diagnostics(
    status: ExecutionStatus,
) -> None:
    executions = _ResultExecutions()
    executions.record.status = status
    executions.record.error_code = (
        ErrorCode.INTERNAL_ERROR.value if status is ExecutionStatus.FAILED
        else ErrorCode.EXECUTION_CANCELLED.value
    )
    executions.record.error_diagnostics = (
        {"exception_type": "unvalidated"} if status is ExecutionStatus.FAILED
        else ErrorDiagnostics.from_exception(RuntimeError("invalid cancellation evidence"))
    )
    executions.result.output = None
    history = RuntimeHistory(
        SimpleNamespace(), tenant_id="tenant", executions=executions,
        authorization=TenantAuthorizationPolicy("tenant"),
    )
    with pytest.raises(AIError) as error:
        await history.result(
            "execution", principal=Principal("caller", "tenant", "service"),
        )
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_runtime_history_inspection_uses_safe_durable_summaries() -> None:
    executions = _ResultExecutions()
    history = RuntimeHistory(
        _InspectionService(),  # type: ignore[arg-type]
        tenant_id="tenant",
        executions=executions,  # type: ignore[arg-type]
        authorization=TenantAuthorizationPolicy("tenant"),
    )
    info = await history.inspect_execution(
        "execution",
        principal=Principal("caller", "tenant", "service"),
    )

    assert info.started_at == executions.record.started_at
    assert info.terminal_at == executions.result.created_at
    assert info.terminal_at != executions.record.updated_at
    assert info.binding_digest == executions.binding.binding_digest
    assert info.usage == UsageSummary(
        logical_requests=1,
        succeeded_requests=1,
        input_tokens=2,
        output_tokens=3,
        model_duration_ns=4,
    )
    assert info.error_diagnostics is None


class _TaskResults:
    def __init__(self) -> None:
        result_digest = canonical_sha256(None)
        node = TaskNode("node")
        state = TaskNodeView(
            "graph",
            "node",
            (),
            TaskStatus.SUCCEEDED,
            None,
            1,
            None,
            result_digest,
            None,
            None,
            "execution",
        )
        self._graph_state = TaskGraphState(
            "graph",
            TaskStatus.SUCCEEDED,
            TaskGraph("graph", (node,)).nodes,
            (state,),
        )
        self.record = TaskResultRecord(
            "graph",
            "node",
            result_digest,
            execution_id="execution",
        )

    async def get_header(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if graph_id == "graph" and tenant_id == "tenant":
            return ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id)
        return None

    async def graph_state(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphState | None:
        if graph_id == "graph" and tenant_id == "tenant":
            return self._graph_state
        return None

    async def get_results(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> dict[str, TaskResultRecord]:
        if graph_id == "graph" and tenant_id == "tenant" and node_ids == ("node",):
            return {"node": self.record}
        return {}


class _Artifacts:
    async def list(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: str | None = None,
        limit: int = 100,
    ) -> Page[ArtifactView]:
        assert execution_id == "execution"
        assert principal.tenant_id == "tenant"
        assert cursor is None
        assert limit == 10
        return Page((ArtifactView("artifact", execution_id, 3),))


@pytest.mark.asyncio
async def test_runtime_history_reads_execution_task_results_and_artifacts() -> None:
    executions = _ResultExecutions()
    tasks = _TaskResults()
    history = RuntimeHistory(
        SimpleNamespace(),
        tenant_id="tenant",
        executions=executions,  # type: ignore[arg-type]
        tasks=tasks,  # type: ignore[arg-type]
        authorization=TenantAuthorizationPolicy("tenant"),
        namespace="workspace",
        artifacts=_Artifacts(),  # type: ignore[arg-type]
    )
    principal = Principal("caller", "tenant", "service")

    result = await history.result("execution", principal=principal)
    reference = await history.task_result_ref(
        "graph",
        "node",
        principal=principal,
    )
    task_result = await history.task_result(
        "graph",
        "node",
        principal=principal,
    )
    artifacts = await history.artifacts(
        "execution",
        principal=principal,
        limit=10,
    )

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.output is None
    assert reference.namespace == "workspace"
    assert reference.tenant_id == "tenant"
    assert reference.graph_id == "graph"
    assert reference.node_id == "node"
    assert reference.result_digest == canonical_sha256(None)
    assert task_result is None
    assert artifacts.items == (ArtifactView("artifact", "execution", 3),)


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ("history-result", "history-ref", "live-result", "live-ref"))
@pytest.mark.parametrize(
    "corruption",
    ("execution", "digest", "missing-result", "missing-execution", "missing-digest", "none"),
)
async def test_task_result_requires_matching_durable_execution(
    api: str,
    corruption: str,
) -> None:
    principal = Principal("caller", "tenant", "service")
    tasks = _TaskResults()
    executions = _ResultExecutions()
    execution_reads: list[str] = []
    if corruption == "execution":
        tasks.record = replace(tasks.record, execution_id="other-execution")
    elif corruption == "digest":
        tasks.record = replace(tasks.record, result_digest=canonical_sha256("other"))
    elif corruption == "missing-digest":
        tasks._graph_state = replace(
            tasks._graph_state,
            node_states=(replace(tasks._graph_state.node_states[0], result_digest=None),),
        )
    elif corruption == "missing-execution":
        tasks._graph_state = replace(
            tasks._graph_state,
            node_states=(replace(tasks._graph_state.node_states[0], execution_id=None),),
        )

    async def get_results(
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> dict[str, TaskResultRecord]:
        assert (graph_id, node_ids, tenant_id) == ("graph", ("node",), "tenant")
        return {} if corruption == "missing-result" else {"node": tasks.record}

    async def get_header(execution_id: str, *, tenant_id: str) -> ResourceRef:
        assert tenant_id == "tenant"
        execution_reads.append(execution_id)
        return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)

    async def get(execution_id: str, *, tenant_id: str) -> object:
        assert tenant_id == "tenant"
        execution_reads.append(execution_id)
        return SimpleNamespace(**{
            **vars(executions.record),
            "execution_id": execution_id,
            "root_execution_id": execution_id,
        })

    async def get_result(execution_id: str, *, tenant_id: str) -> object:
        assert tenant_id == "tenant"
        execution_reads.append(execution_id)
        return executions.result

    tasks.get_results = get_results
    executions.get_header = get_header
    executions.get = get
    executions.get_result = get_result
    history = RuntimeHistory(
        SimpleNamespace(),
        tenant_id="tenant",
        namespace="workspace",
        executions=executions,  # type: ignore[arg-type]
        tasks=tasks,  # type: ignore[arg-type]
        authorization=TenantAuthorizationPolicy("tenant"),
    )

    async def graph_state(graph_id: str, *, principal: Principal) -> TaskGraphState:
        assert (graph_id, principal.tenant_id) == ("graph", "tenant")
        return tasks._graph_state

    async def get_result_record(
        graph_id: str, node_id: str, *, tenant_id: str
    ) -> TaskResultRecord | None:
        return (await get_results(graph_id, (node_id,), tenant_id=tenant_id)).get(node_id)

    async def read_result_record(record: TaskResultRecord, *, principal: Principal) -> object:
        return (await history.result(record.execution_id, principal=principal)).output

    task_runtime = SimpleNamespace(
        get_result_record=get_result_record,
        read_result_record=read_result_record,
    )
    live = TaskGraphRun(
        SimpleNamespace(
            namespace="workspace",
            _require_task_node_runtime=lambda: task_runtime,
        ),
        SimpleNamespace(state=graph_state),
        "graph",
        principal,
        None,
    )
    if api == "history-result":
        request = history.task_result("graph", "node", principal=principal)
    elif api == "history-ref":
        request = history.task_result_ref("graph", "node", principal=principal)
    elif api == "live-result":
        request = live.result("node")
    else:
        request = live.result_ref("node")
    if corruption == "none":
        result = await request
        if api.endswith("-result"):
            assert result is None
            assert execution_reads and set(execution_reads) == {"execution"}
        else:
            assert result.result_digest == canonical_sha256(None)
            assert execution_reads == []
    else:
        with pytest.raises(AIError) as raised:
            await request
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert raised.value.safe_details == {}
        assert execution_reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ("history-result", "history-ref", "live-result", "live-ref"))
@pytest.mark.parametrize("status", (
    TaskStatus.READY, TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED,
))
async def test_task_result_preserves_unready_and_failed_error_details(
    api: str,
    status: TaskStatus,
) -> None:
    principal = Principal("caller", "tenant", "service")
    tasks = _TaskResults()
    tasks._graph_state = replace(
        tasks._graph_state,
        node_states=(replace(tasks._graph_state.node_states[0], status=status, error_code="failed"),),
    )
    history = RuntimeHistory(
        SimpleNamespace(), tenant_id="tenant", namespace="workspace",
        tasks=tasks, authorization=TenantAuthorizationPolicy("tenant"),
    )

    async def graph_state(graph_id: str, *, principal: Principal) -> TaskGraphState:
        return tasks._graph_state

    live = TaskGraphRun(
        SimpleNamespace(namespace="workspace"), SimpleNamespace(state=graph_state),
        "graph", principal, None,
    )
    if api.startswith("history"):
        read = history.task_result_ref if api.endswith("ref") else history.task_result
        request = read("graph", "node", principal=principal)
    else:
        read = live.result_ref if api.endswith("ref") else live.result
        request = read("node")
    with pytest.raises(AIError) as raised:
        await request
    if status is TaskStatus.READY:
        assert raised.value.code is ErrorCode.TASK_NOT_READY
    else:
        assert raised.value.code is ErrorCode.TASK_NODE_FAILED
        assert raised.value.safe_details == {
            "graph_id": "graph", "node_id": "node", "status": status.value, "error_code": "failed",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", (False, True))
async def test_history_task_result_authorizes_before_reading_records(reference: bool) -> None:
    class DeniedTasks(_TaskResults):
        async def graph_state(self, *args: object, **kwargs: object) -> TaskGraphState:
            raise AssertionError("unauthorized graph must not be read")

        async def get_results(self, *args: object, **kwargs: object) -> dict[str, TaskResultRecord]:
            raise AssertionError("unauthorized result must not be read")

    history = RuntimeHistory(
        SimpleNamespace(), tenant_id="tenant", namespace="workspace",
        tasks=DeniedTasks(), authorization=TenantAuthorizationPolicy("tenant"),
    )
    read = history.task_result_ref if reference else history.task_result
    with pytest.raises(AIError) as raised:
        await read("graph", "node", principal=Principal("caller", "other-tenant", "service"))
    assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ("history", "trace"))
async def test_execution_query_cursor_binds_exact_selector(query: str) -> None:
    service = DefaultExecutionHistoryService(
        _Executions(), TenantAuthorizationPolicy("tenant"), _PagingReader(),
        HmacCursorSigner("public-history", b"public-history-key"),
    )
    principal = Principal("caller", "tenant", "service")
    read = getattr(service, query)
    filters = {
        "agent_run_seq": 1,
        "model_request_seq": 2,
        "step_index": 0,
        "tool_call_id": "call",
    }
    if query == "history":
        filters.update(message_seq=3, part_index=0)
    page = await read("execution", principal=principal, **filters)
    assert page.next_cursor is not None
    for name, value in filters.items():
        for replacement in (None, "other" if isinstance(value, str) else value + 1):
            changed = {**filters, name: replacement}
            with pytest.raises(AIError) as error:
                await read(
                    "execution", principal=principal, cursor=page.next_cursor, **changed,
                )
            assert error.value.code is ErrorCode.CURSOR_INVALID
    if query == "history":
        with pytest.raises(AIError) as error:
            await read(
                "execution", principal=principal, cursor=page.next_cursor,
                include_content=True, **filters,
            )
        assert error.value.code is ErrorCode.CURSOR_INVALID
    second = await read("execution", principal=principal, cursor=page.next_cursor, **filters)
    assert (second.items[0].message_seq if query == "history" else second.items[0].step_event_seq) == 1
    assert second.next_cursor is None


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ("service", "history", "executions", "execution"))
@pytest.mark.parametrize("include_content", (False, True))
async def test_execution_history_and_trace_forward_filters(
    entrypoint: str, include_content: bool,
) -> None:
    reader = _Reader()
    service = DefaultExecutionHistoryService(
        _Executions(), TenantAuthorizationPolicy("tenant"), reader,
    )
    principal = Principal("caller", "tenant", "service")
    executions = RuntimeExecutions(service, None)
    execution = Execution(SimpleNamespace(executions=executions), "execution", principal, None)
    target = {
        "service": service,
        "history": RuntimeHistory(service, tenant_id="tenant"),
        "executions": executions,
        "execution": execution,
    }[entrypoint]
    args = () if entrypoint == "execution" else ("execution",)
    kwargs = {} if entrypoint == "execution" else {"principal": principal}
    trace_filters = {
        "agent_run_seq": 2,
        "model_request_seq": 3,
        "step_index": 0,
        "tool_call_id": "call",
    }
    history_filters = {**trace_filters, "message_seq": 4, "part_index": 0}
    history = await target.history(
        *args, include_content=include_content, **kwargs, **history_filters,
    )
    trace = await target.trace(
        *args, **kwargs, **trace_filters,
    )
    assert reader.history_filters == history_filters
    assert reader.trace_filters == trace_filters
    item = history.items[0]
    assert (
        item.execution_id, item.agent_run_seq, item.model_request_seq,
        item.step_index, item.message_seq, item.part_index, item.tool_call_id,
    ) == ("execution", 2, 3, 0, 4, 0, "call")
    assert item.content_included is include_content
    assert item.content == ("hello" if include_content else None)
    assert trace.items[0].payload == {"kind": "TEST"}


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ("history", "trace"))
async def test_execution_query_accepts_independent_filters(query: str) -> None:
    reader = _Reader()
    service = DefaultExecutionHistoryService(
        _Executions(), TenantAuthorizationPolicy("tenant"), reader,
    )
    principal = Principal("caller", "tenant", "service")
    read = getattr(service, query)
    filters = {
        "agent_run_seq": 2,
        "model_request_seq": 3,
        "step_index": 0,
        "tool_call_id": "call",
    }
    if query == "history":
        filters.update(message_seq=4, part_index=0)
    for name, value in filters.items():
        page = await read("execution", principal=principal, **{name: value})
        assert len(page.items) == 1
        received = reader.history_filters if query == "history" else reader.trace_filters
        assert received == {field: value if field == name else None for field in filters}


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ("history", "trace"))
async def test_execution_query_rejects_invalid_filter_values(query: str) -> None:
    service = DefaultExecutionHistoryService(
        _Executions(), TenantAuthorizationPolicy("tenant"), _Reader(),
    )
    principal = Principal("caller", "tenant", "service")
    read = getattr(service, query)
    integer_filters = {"agent_run_seq": 1, "model_request_seq": 1, "step_index": 0}
    if query == "history":
        integer_filters.update(message_seq=1, part_index=0)
    invalid = [
        (name, value)
        for name, minimum in integer_filters.items()
        for value in (True, 1.5, "1", minimum - 1)
    ]
    invalid.extend((("tool_call_id", ""), ("tool_call_id", 1)))
    for name, value in invalid:
        with pytest.raises(AIError) as error:
            await read("execution", principal=principal, **{name: value})
        assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert error.value.safe_details == {"field": name}
