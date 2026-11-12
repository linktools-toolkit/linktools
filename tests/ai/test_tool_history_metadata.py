#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""History reads canonical effect identity without hydrating tool receipts."""

from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import ToolOperationStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive import ContentArchive
from linktools.ai.persistence._step_store import SqlStepStore
from .test_tool_receipt_lookup_io import _complete, _identity


@pytest.mark.asyncio
async def test_tool_history_metadata_uses_one_owner_read_and_no_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "history.db")
    async with Persistence.open("native-effects", tenant_id="tenant", storage=storage) as state:
        await _complete(state, "run", "call")
    async with Persistence.open("native-effects", tenant_id="tenant", storage=storage, read_only=True) as state:
        async def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("history identity must not hydrate tool payloads")
        monkeypatch.setattr(ContentArchive, "read_contents", forbidden)
        monkeypatch.setattr(SqlStepStore, "capture_message_roots", forbidden)
        repository = state.checkpoint.tools
        statements = []
        def capture(_connection: object, _cursor: object, statement: str, *_args: object) -> None:
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(statement)
        engine = repository.state_store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", capture)
        try:
            identity = _identity("run", "call")
            values = await repository.get_operation_metadata((identity, identity, "missing"), tenant_id="tenant")
            assert len(values) == 1
            value, = values
            assert (value.tool_operation_id, value.execution_id, value.tool_call_id, value.status) == (
                identity, "execution", "call", ToolOperationStatus.COMPLETED)
            assert len(statements) == 1
            statements.clear()
            assert await repository.get_operation_metadata((), tenant_id="tenant") == ()
            assert await repository.get_operation_metadata((identity,), tenant_id="other") == ()
            assert not statements
        finally:
            event.remove(engine, "before_cursor_execute", capture)
