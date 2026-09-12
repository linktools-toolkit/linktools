#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public execution metadata query coverage."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    Principal,
    ResourceKind,
    ResourceRef,
    TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ListExecutionRequest, RuntimeHistory
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime._history_service import DefaultExecutionHistoryService
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._contracts import (
    ExecutionCandidate,
    ExecutionCandidatePage,
    ExecutionRecord,
    StoredUserInput,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StoredPayload


class _Reader:
    pass


class _HistoryService:
    def __init__(self) -> None:
        self.request = None

    async def list(self, request: ListExecutionRequest) -> str:
        self.request = request
        return "page"


class _DenyOne:
    async def authorize(self, principal, action, resource) -> None:
        del action
        if resource.id == "exec-002":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if principal.tenant_id != resource.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


class _CandidateRepository:
    def __init__(self, records: tuple[ExecutionRecord, ...]) -> None:
        self.records = records
        self.calls: list[int] = []

    async def list_candidates(
        self,
        *,
        tenant_id: str,
        session_id: str | None,
        parent_execution_id: str | None,
        cursor: str | None,
        limit: int,
    ) -> ExecutionCandidatePage:
        del session_id, parent_execution_id
        assert tenant_id == "tenant"
        self.calls.append(limit)
        start = 0 if cursor is None else int(cursor) + 1
        selected = self.records[start : start + limit]
        return ExecutionCandidatePage(
            tuple(
                ExecutionCandidate(record, str(start + index))
                for index, record in enumerate(selected)
            ),
            start + limit < len(self.records),
        )

    async def get_header(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if tenant_id != "tenant":
            return None
        return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)


def _binding(agent_id: str) -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        agent_spec=AgentSpec(agent_id, model="model"),
        base_model={"route_id": "model", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


def _record(
    execution_id: str,
    *,
    agent_id: str,
    session_id: str | None = None,
    parent_execution_id: str | None = None,
    source_execution_id: str | None = None,
    base_execution_id: str | None = None,
) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    is_child = parent_execution_id is not None
    return ExecutionRecord(
        execution_id=execution_id,
        tenant_id="tenant",
        session_id=session_id,
        parent_execution_id=parent_execution_id,
        root_execution_id="exec-001" if is_child else execution_id,
        source_execution_id=source_execution_id,
        base_execution_id=base_execution_id,
        lineage_kind=(
            ExecutionLineageKind.SUBAGENT
            if is_child
            else ExecutionLineageKind.RUN
        ),
        status=ExecutionStatus.SUCCEEDED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=1,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(agent_id),
        principal_id="owner",
        principal_kind="service",
        stored_user_input=StoredUserInput(
            "text",
            StoredPayload.inline_text("prompt"),
        ),
        parent_invocation_id="call" if is_child else None,
    )


def _service(repository) -> DefaultExecutionHistoryService:
    return DefaultExecutionHistoryService(
        repository,
        TenantAuthorizationPolicy("tenant"),
        _Reader(),
        HmacCursorSigner("execution", b"query-key"),
    )


@pytest.mark.asyncio
async def test_execution_service_list_delegates_public_query_request() -> None:
    service = object.__new__(DefaultExecutionService)
    history_service = _HistoryService()
    service._history_service = history_service
    request = ListExecutionRequest(Principal("caller", "tenant", "service"))

    assert await service.list(request) == "page"
    assert history_service.request is request


@pytest.mark.asyncio
async def test_execution_list_applies_tenant_filters_and_direct_parent() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="execution-query", tenant_id="tenant")
    try:
        records = (
            _record("exec-001", agent_id="agent-a", session_id="session-1"),
            _record("exec-002", agent_id="agent-b", session_id="session-1"),
            _record("exec-003", agent_id="agent-a"),
            _record(
                "exec-004",
                agent_id="agent-b",
                session_id="session-1",
                parent_execution_id="exec-001",
            ),
            _record(
                "exec-005",
                agent_id="agent-a",
                source_execution_id="exec-001",
                base_execution_id="exec-001",
            ),
        )
        for record in records:
            await state.execution.executions.create(record)
        service = _service(state.execution.executions)
        principal = Principal("caller", "tenant", "service")

        tenant_page = await service.list(ListExecutionRequest(principal))
        assert {item.execution_id for item in tenant_page.items} == {
            record.execution_id for record in records
        }
        assert tenant_page.items[2].session_id is None

        session_page = await service.list(
            ListExecutionRequest(principal, session_id="session-1")
        )
        assert {item.execution_id for item in session_page.items} == {
            "exec-001",
            "exec-002",
            "exec-004",
        }

        combined = await service.list(
            ListExecutionRequest(
                principal,
                session_id="session-1",
                agent_id="agent-b",
                parent_execution_id="exec-001",
            )
        )
        assert [item.execution_id for item in combined.items] == ["exec-004"]

        agent_page = await service.list(
            ListExecutionRequest(principal, agent_id="agent-a")
        )
        assert {item.execution_id for item in agent_page.items} == {
            "exec-001",
            "exec-003",
            "exec-005",
        }

        parent_page = await service.list(
            ListExecutionRequest(principal, parent_execution_id="exec-001")
        )
        assert [item.execution_id for item in parent_page.items] == ["exec-004"]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_execution_list_cursor_binds_identity_and_continues_without_repeat() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="execution-cursor", tenant_id="tenant")
    try:
        records = tuple(
            _record(
                f"exec-{index:03d}",
                agent_id="agent-a" if index % 2 else "agent-b",
            )
            for index in range(1, 6)
        )
        for record in records:
            await state.execution.executions.create(record)
        service = _service(state.execution.executions)
        principal = Principal("caller", "tenant", "service")
        request = ListExecutionRequest(principal, limit=2)

        first = await service.list(request)
        assert first.next_cursor is not None
        pages = [first]
        cursor = first.next_cursor
        while cursor is not None:
            page = await service.list(replace(request, cursor=cursor))
            pages.append(page)
            cursor = page.next_cursor
        ids = [item.execution_id for page in pages for item in page.items]
        assert len(ids) == len(set(ids)) == len(records)

        for changed in (
            replace(request, cursor=first.next_cursor, agent_id="agent-a"),
            replace(request, cursor=first.next_cursor, session_id="session-1"),
            replace(
                request,
                cursor=first.next_cursor,
                parent_execution_id="exec-001",
            ),
            replace(
                request,
                cursor=first.next_cursor,
                principal=Principal("other", "tenant", "service"),
            ),
            replace(
                request,
                cursor=first.next_cursor,
                principal=Principal("caller", "other-tenant", "service"),
            ),
        ):
            with pytest.raises(AIError) as error:
                await service.list(changed)
            assert error.value.code is ErrorCode.CURSOR_INVALID
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_execution_list_skips_unauthorized_candidates() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="execution-auth", tenant_id="tenant")
    try:
        for index in range(1, 4):
            await state.execution.executions.create(
                _record(f"exec-{index:03d}", agent_id="agent")
            )
        service = DefaultExecutionHistoryService(
            state.execution.executions,
            _DenyOne(),
            _Reader(),
            HmacCursorSigner("execution", b"query-key"),
        )
        page = await service.list(
            ListExecutionRequest(Principal("caller", "tenant", "service"))
        )
        assert [item.execution_id for item in page.items] == [
            "exec-001",
            "exec-003",
        ]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_execution_list_skips_new_records_before_cursor() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="execution-weak-consistency", tenant_id="tenant")
    try:
        for execution_id in ("exec-001", "exec-002", "exec-003"):
            await state.execution.executions.create(
                _record(execution_id, agent_id="agent")
            )
        service = _service(state.execution.executions)
        principal = Principal("caller", "tenant", "service")
        request = ListExecutionRequest(principal, limit=1)

        first = await service.list(request)
        assert first.next_cursor is not None
        await state.execution.executions.create(
            _record("exec-000", agent_id="agent")
        )

        second = await service.list(replace(request, cursor=first.next_cursor))
        assert [item.execution_id for item in second.items] == [
            "exec-002",
        ]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_execution_list_exact_physical_boundary_has_no_false_cursor() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="execution-boundary", tenant_id="tenant")
    try:
        for index in range(1000):
            await state.execution.executions.create(
                _record(
                    f"exec-{index:04d}",
                    agent_id="other",
                    session_id="bounded",
                )
            )
        service = _service(state.execution.executions)
        page = await service.list(
            ListExecutionRequest(
                Principal("caller", "tenant", "service"),
                session_id="bounded",
                agent_id="target",
            )
        )

        assert page.items == ()
        assert page.next_cursor is None
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_runtime_and_history_execution_queries_share_projection() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="execution-parity", tenant_id="tenant")
    try:
        record = _record(
            "exec-001",
            agent_id="agent",
            session_id="session-1",
        )
        await state.execution.executions.create(record)
        service = _service(state.execution.executions)
        execution_service = object.__new__(DefaultExecutionService)
        execution_service._history_service = service
        history = RuntimeHistory(service, tenant_id="tenant")
        request = ListExecutionRequest(
            Principal("caller", "tenant", "service"),
            session_id="session-1",
        )

        assert await execution_service.list(request) == await history.list_executions(
            request
        )
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_execution_list_stops_at_physical_candidate_limit() -> None:
    records = tuple(
        _record(
            f"exec-{index:04d}",
            agent_id="target" if index == 1000 else "other",
        )
        for index in range(1001)
    )
    repository = _CandidateRepository(records)
    service = _service(repository)
    request = ListExecutionRequest(
        Principal("caller", "tenant", "service"),
        agent_id="target",
    )

    first = await service.list(request)
    assert first.items == ()
    assert first.next_cursor is not None
    assert repository.calls == [1000]

    second = await service.list(replace(request, cursor=first.next_cursor))
    assert [item.execution_id for item in second.items] == ["exec-1000"]
    assert repository.calls == [1000, 1000]


def test_list_execution_request_rejects_limits_outside_public_range() -> None:
    principal = Principal("caller", "tenant", "service")
    for limit in (0, 201):
        with pytest.raises(AIError) as error:
            ListExecutionRequest(principal, limit=limit)
        assert error.value.code is ErrorCode.PAGE_LIMIT_INVALID
