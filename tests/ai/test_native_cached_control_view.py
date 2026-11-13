#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unconsumed Native bodies do not invalidate bounded control observations."""
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.core import Principal
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._execution_content import ExecutionContent
from linktools.ai.runtime import Runtime
from linktools.ai.storage import InMemoryCacheBackend, _write_cache

from .test_native_text_result_runtime import _prepare, _publish
from .test_budget_usage_projection import _LargeModels
from .test_run_budget_public_api import _capabilities
from ._runtime_test_helpers import model_features


@pytest.mark.asyncio
async def test_native_control_view_and_budget_use_cache_without_body_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_write_cache, "time", lambda: 100.0)
    config = PersistenceStorage.sqlite(tmp_path / "cached.db", cache=InMemoryCacheBackend(max_bytes=2_000_000))
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare(state, "native result", "native result")
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        statements = []
        engine = repository.state_store.storage_group.context.engine.sync_engine

        def capture(_connection, _cursor, statement, *_args):
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(statement)

        async def forbid(*args, **kwargs):
            raise AssertionError("control view hydrated unused Native bodies")

        event.listen(engine, "before_cursor_execute", capture)
        try:
            with monkeypatch.context() as patch:
                patch.setattr(ExecutionContent, "read_payloads", forbid)
                expected = await repository.get_view("execution", tenant_id="tenant")
                assert expected is not None
                statements.clear()
                assert await repository.get_view("execution", tenant_id="tenant") == expected
                assert statements == []
                async with Runtime.attach(persistence=state, features=model_features(_LargeModels(), _capabilities())) as runtime:
                    assert await runtime.executions.budget_usage("execution", principal=Principal(current.principal_id, "tenant", current.principal_kind)) is None
                    statements.clear()
                    assert await runtime.executions.budget_usage("execution", principal=Principal(current.principal_id, "tenant", current.principal_kind)) is None
                    assert statements == []
                assert await repository.get_view("execution", tenant_id="foreign") is None
            # Fresh still uses a real source snapshot even when the ordinary view is cached.
            statements.clear()
            assert await repository.get_view_fresh("execution", tenant_id="tenant") == expected
            assert statements
            key = repository.execution_record_key("execution")
            await repository.state_store.mutate(lambda tx: tx.delete_record(key))
            assert await repository.get_view_fresh("execution", tenant_id="tenant") is None
        finally:
            event.remove(engine, "before_cursor_execute", capture)


@pytest.mark.asyncio
async def test_native_control_view_rejects_missing_consumed_binding(tmp_path: Path) -> None:
    from linktools.ai.persistence.state import RecordQuery

    config = PersistenceStorage.sqlite(tmp_path / "missing.db", cache=InMemoryCacheBackend(max_bytes=2_000_000))
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare(state, "native result", "native result")
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        assert await repository.get_view("execution", tenant_id="tenant") is not None
        archive = await repository._execution_content._archive()
        rows = await archive.store.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
        owner = next(row for row in rows if row.data["codec"] == "execution-binding")
        await archive.store.mutate(lambda tx: tx.delete_record(owner.key))
        with pytest.raises(AIError) as raised:
            await repository.get_view_fresh("execution", tenant_id="tenant")
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
