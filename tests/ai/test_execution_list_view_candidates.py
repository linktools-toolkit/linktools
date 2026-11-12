#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""State lists consume typed views, not execution input/result bodies."""
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import Principal
from linktools.ai.execution import ListExecutionRequest, StoredUserInput, project_execution_view
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._execution_content import ExecutionContent
from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.storage import StoredPayload

from .test_runtime_execution_query import _record, _service
from .test_runtime_persistence_evidence import _result


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_state_list_pages_without_hydrating_execution_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "views.db")
    async with Persistence.open("execution-query", tenant_id="tenant", storage=storage) as state:
        repository = state.execution.executions
        records = [replace(_record(f"exec-{index}", agent_id="agent", session_id="session"),
                           stored_user_input=StoredUserInput("text", StoredPayload.inline_text("large input " * 10000)))
                   for index in range(3)]
        records[0] = replace(records[0], completed_at=records[0].updated_at,
            result=replace(_result(records[0].execution_id, "inline", records[0].updated_at),
                           output=StoredPayload.inline_json({"text": "large result " * 10000})))
        for record in records:
            await repository.create(record)

        async def reject_payloads(*args, **kwargs):
            raise AssertionError("state list must not hydrate input/output payloads")

        monkeypatch.setattr(ExecutionContent, "read_payloads", reject_payloads)
        monkeypatch.setattr(ExecutionRepositoryImpl, "_decode_many", reject_payloads)
        service = _service(repository)
        request = ListExecutionRequest(Principal("caller", "tenant", "service"), session_id="session", limit=1)
        actual = []
        while True:
            page = await service.list(request)
            actual.extend(page.items)
            if page.next_cursor is None:
                break
            request = replace(request, cursor=page.next_cursor)
        assert {view.execution_id: view for view in actual} == {
            record.execution_id: project_execution_view(record) for record in records}
        assert len(actual) == 3
