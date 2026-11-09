#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Additional public contract verification for SQL step storage."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelMessagesTypeAdapter, ModelRequest, ModelResponse, TextPart, ToolReturnPart
from pydantic_ai_harness.media import DiskMediaStore
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.persistence import SqlStepStore
from linktools.ai.errors import AIError


@pytest.mark.asyncio
async def test_independent_engines_publish_prune_and_replay_consistently(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'concurrent.db'}"
    engines = [create_async_engine(url), create_async_engine(url)]
    stores = [SqlStepStore(engine, namespace="same", tenant_id="same", max_snapshots_per_run=1) for engine in engines]
    shared = ModelResponse(parts=[TextPart("shared")])
    snapshots = [ContinuableSnapshot(run_id="same", step_index=i, idempotency_key=f"{i}:complete",
                                    messages=[shared, ModelResponse(parts=[TextPart(str(i))])])
                 for i in range(12)]
    try:
        await stores[0].initialize(local=True)
        await asyncio.gather(*(stores[i % 2].save_snapshot(snapshot) for i, snapshot in enumerate(snapshots)))
        terminal = replace(snapshots[-1], step_index=99, idempotency_key="terminal")
        await stores[0].save_snapshot(terminal)
        assert await stores[1].list_snapshots(run_id="same") == [terminal]
        await asyncio.gather(*(stores[i % 2].save_snapshot(snapshot) for i, snapshot in enumerate(snapshots)))
        assert await stores[0].latest_snapshot(run_id="same") == terminal
        async with engines[0].connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_sequences")) == 13
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_snapshots WHERE pruned = 0 AND sequence_root IS NOT NULL")) == 1
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_snapshots")) == 13
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_snapshots WHERE pruned = 1 AND messages = '{}' AND sequence_root IS NULL")) == 12
    finally:
        for store in stores:
            await store.close()
        for engine in engines:
            await engine.dispose()


@pytest.mark.asyncio
async def test_externalized_nested_tool_files_keep_file_provenance(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'media.db'}")
    store = SqlStepStore(engine, namespace="media", tenant_id="one", media_store=DiskMediaStore(directory=tmp_path / 'files'))
    content = {"file": BinaryContent(data=b"\xff\xfe" * 40000, media_type="application/octet-stream"),
               "ordinary": {"kind": "image-url", "url": "https://example.test/caller-data.png"},
               "literal": {"__harness_external_media__": "caller data", "content": "unaltered"}}
    original = ContinuableSnapshot(run_id="run", step_index=0, messages=[ModelRequest(parts=[ToolReturnPart("work", content, tool_call_id="call")])])
    try:
        await store.initialize(local=True)
        await store.save_snapshot(original)
        recovered = await store.latest_snapshot(run_id="run")
        assert isinstance(recovered.messages[0].parts[0].content["file"], BinaryContent)
        assert isinstance(recovered.messages[0].parts[0].content["ordinary"], dict)
        assert json.loads(ModelMessagesTypeAdapter.dump_json(recovered.messages)) == json.loads(ModelMessagesTypeAdapter.dump_json(original.messages))
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_bound_clone_cannot_reopen_owner_after_owner_closes(tmp_path: Path) -> None:
    from pydantic_ai_harness.step_persistence import InMemoryStepStore

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'lifecycle.db'}")
    store = SqlStepStore(engine, namespace="same", tenant_id="same")
    try:
        clone = store.with_tool_effect_store(InMemoryStepStore())
        await store.initialize(local=True)
        await clone.register_run(RunRecord(run_id="one"))
        await store.close()
        with pytest.raises(AIError):
            await clone.get_run(run_id="one")
    finally:
        await store.close()
        await engine.dispose()
