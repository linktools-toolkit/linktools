#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending admission reads do not hydrate an unrelated terminal outcome."""

from pathlib import Path

import pytest

from linktools.ai.core import ToolOperationStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive import ContentArchive
from linktools.ai.persistence._pending_operation_reference import read_pending_operations
from linktools.ai.persistence.contracts import ToolOperationRecord
from linktools.ai.storage import StoredPayload
from .test_step_projection import _execution


@pytest.mark.asyncio
async def test_pending_projection_reads_only_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    execution = _execution()
    arguments = StoredPayload.inline_json({"argument": True})
    output = StoredPayload.inline_json({"unrelated": "terminal result"})
    operation = ToolOperationRecord("operation", "execution", "run", "call", "tool", arguments.digest,
        "binding", True, ToolOperationStatus.COMPLETED, None, 1, None,
        execution.created_at, execution.updated_at, arguments_payload=arguments, result_payload=output)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage) as state:
        await state.checkpoint.tools.reserve(operation)
    calls = []
    original = ContentArchive.read_contents

    async def read(archive: ContentArchive, identities: tuple, **kwargs: object) -> tuple:
        calls.append(tuple(identities))
        return await original(archive, identities, **kwargs)

    monkeypatch.setattr(ContentArchive, "read_contents", read)
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage, read_only=True) as state:
        values = await read_pending_operations(state.checkpoint.tools.state_store, ("operation",),
            namespace="native-result", tenant_id="tenant")
        assert values[0].arguments_payload == arguments
        assert values[0].execution_id == "execution"
    assert len(calls) == 1
    assert len(calls[0]) == 1
