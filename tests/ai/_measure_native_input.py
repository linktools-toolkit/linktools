#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone Native input/output batching and metadata SQL measurement."""

import asyncio
import json
import tempfile
from collections import Counter
from pathlib import Path

from sqlalchemy import event
from linktools.ai.model import StoredUserInput
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.storage import StoredPayload
from tests.ai.test_native_input_runtime import _prepare_input
from tests.ai.test_native_text_result_runtime import _publish


async def main() -> None:
    config = PersistenceStorage.sqlite(Path(tempfile.mkdtemp(prefix="native-input-")) / "state.db")
    body = "terminal body " * 10000
    original = StoredUserInput("text", StoredPayload.inline_text(body))
    counts: Counter[str] = Counter()
    result: dict[str, object] = {"input_bytes": original.payload.size}
    def sql(connection: object, cursor: object, statement: str, parameters: object,
            context: object, executemany: bool) -> None:
        counts[statement.split()[0]] += 1
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare_input(state, original, body, body)
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        engine = repository.state_store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", sql)
        try:
            loaded = await repository.get_fresh("execution", tenant_id="tenant")
            assert loaded.stored_user_input == original and loaded.result == commit.result
            result["warm_full_input_and_output"] = dict(counts)
            counts.clear()
            await repository.get_metadata_record("execution", tenant_id="tenant")
            result["metadata"] = dict(counts)
            counts.clear()
            await repository.get_result_record("execution", tenant_id="tenant", fresh=True)
            result["result_only"] = dict(counts)
        finally:
            event.remove(engine, "before_cursor_execute", sql)
    async with Persistence.open("native-result", tenant_id="tenant", storage=config, read_only=True) as state:
        repository = state.execution.executions
        engine = repository.state_store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", sql)
        try:
            counts.clear()
            loaded = await repository.get_fresh("execution", tenant_id="tenant")
            assert loaded.stored_user_input == original and loaded.result == commit.result
            result["cold_full_input_and_output"] = dict(counts)
        finally:
            event.remove(engine, "before_cursor_execute", sql)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
