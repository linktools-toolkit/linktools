#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recent diagnostics preserve every Info field without hydrating payloads."""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from linktools.ai.core import Principal, TenantAuthorizationPolicy
from linktools.ai.execution import StoredUserInput
from linktools.ai.history import History
from linktools.ai.history._history import _project_execution_info
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._execution_content import ExecutionContent
from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.storage import StoredPayload

from .test_runtime_execution_query import _record, _service
from .test_runtime_persistence_evidence import _result


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_recent_execution_metadata_pages_without_payload_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "recent.db")
    async with Persistence.open("execution-query", tenant_id="tenant", storage=storage) as state:
        repository = state.execution.executions
        base = _record("base", agent_id="agent", session_id="session")
        records = [replace(base, execution_id=f"exec-{index}", created_at=base.created_at + timedelta(seconds=index),
            stored_user_input=StoredUserInput("text", StoredPayload.inline_text("input " * 10000)),
            completed_at=base.updated_at,
            result=replace(_result(base.execution_id, "inline", base.updated_at),
                           output=StoredPayload.inline_json({"text": "result " * 10000}))) for index in range(3)]
        for record in records:
            await repository.create(record)
        original = repository._candidate_records
        pages = []

        async def one_page(**kwargs):
            pages.append(kwargs["cursor"])
            return await original(**{**kwargs, "limit": 1})

        async def no_payloads(*args, **kwargs):
            raise AssertionError("recent diagnostics must not restore input/output")

        monkeypatch.setattr(repository, "_candidate_records", one_page)
        monkeypatch.setattr(ExecutionContent, "read_payloads", no_payloads)
        monkeypatch.setattr(ExecutionRepositoryImpl, "_decode_many", no_payloads)
        history = History(_service(repository), tenant_id="tenant", executions=repository,
                          authorization=TenantAuthorizationPolicy("tenant"))
        actual = await history.recent_executions(principal=Principal("caller", "tenant", "service"), limit=2)
        assert actual == tuple(_project_execution_info(record) for record in reversed(records[1:]))
        assert len(pages) == 3 and pages[0] is None and all(pages[1:])
