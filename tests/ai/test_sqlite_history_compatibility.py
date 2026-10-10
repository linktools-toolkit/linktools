#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persisted SQLite history remains usable after optional wire fields are added."""

import base64
import hashlib
import json
import sqlite3
import zlib
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, UserPromptPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus, Principal
from linktools.ai.runtime import Runtime, RuntimeHistory, RuntimeStorage

from . import _runtime_test_helpers as helpers


_FIXTURE = Path(__file__).parent / "fixtures" / "persistence" / "sqlite_session_history_5b7dd032.json"


async def _old_execution_evidence(
    history: RuntimeHistory, execution_id: str, principal: Principal,
) -> dict[str, object]:
    interactions = await history.model_interactions(
        execution_id, principal=principal, include_content=True,
    )
    assert len(interactions.items) == 1
    interaction = interactions.items[0]
    assert interaction.status == "SUCCEEDED"
    assert interaction.model_request_seq == 1
    assert (interaction.agent_run_seq, interaction.step_index, interaction.purpose) == (1, 1, "agent")
    metadata = await history.model_interactions(execution_id, principal=principal)
    assert len(metadata.items) == 1
    assert metadata.items[0].request == {}
    assert metadata.items[0].response is None
    assert metadata.items[0].usage == interaction.usage
    assert interaction.request["messages"]
    assert "retained old question" in json.dumps(interaction.request)
    assert interaction.response is not None
    assert "done" in json.dumps(interaction.response)
    assert interaction.usage is not None
    assert (interaction.usage.input_tokens, interaction.usage.output_tokens) == (101, 202)
    usage = await history.usage(execution_id, principal=principal)
    assert (usage.logical_requests, usage.succeeded_requests) == (1, 1)
    assert (usage.input_tokens, usage.output_tokens) == (101, 202)
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (303, 404)
    trace = await history.trace(
        execution_id, principal=principal, agent_run_seq=1, model_request_seq=1,
    )
    assert [item.step_event_seq for item in trace.items] == [2, 3]
    assert [item.payload["kind"] for item in trace.items] == ["MODEL_REQUEST", "MODEL_RESPONSE"]
    unfiltered_trace = await history.trace(execution_id, principal=principal)
    assert unfiltered_trace.items == trace.items
    assert trace.items[-1].payload["message_seq"] == 2
    messages = await history.history(
        execution_id, principal=principal, include_content=True,
        agent_run_seq=1, model_request_seq=1,
    )
    assert [(item.item_kind, item.content) for item in messages.items] == [("assistant", "done")]
    assert messages.items[0].step_index == 1
    return {"interactions": interactions.items, "usage": usage, "trace": trace.items, "messages": messages.items}


@pytest.mark.asyncio
async def test_sqlite_history_retains_old_turn_when_resumed_and_reopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    sql = zlib.decompress(base64.b64decode(fixture["sql_zlib_base64"]))
    assert hashlib.sha256(sql).hexdigest() == fixture["sql_sha256"]
    assert b'pending_part_count' not in sql
    path = tmp_path / "history.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(sql.decode("utf-8"))
        heads = connection.execute(
            "SELECT payload_json FROM ai_state_records WHERE kind = ?", ("transcript_head",),
        ).fetchall()
        assert heads
        assert all(
            set(json.loads(row[0])["value"]["payload"]["fields"])
            == {"owner_domain", "owner_id", "message_count", "chunk_count", "quality"}
            for row in heads
        )

    observed_prompts: list[str] = []
    original_model = helpers._runtime_usage_model

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        observed_prompts.extend(
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        )
        return await original_model(messages, info)

    monkeypatch.setattr(helpers, "_runtime_usage_model", model)
    application = CapabilityGroup("application")
    application.agent("default", model="default", allow_tools=())
    async with Runtime.open(
        "legacy-history", storage=RuntimeStorage.sqlite(path),
        models=helpers.RuntimeUsageModels(), capabilities=(application,),
    ) as runtime:
        session = runtime.agents.get("default").session("session")
        old = await session.history()
        old_execution_id = (await session.timeline()).items[0].execution_id
        evidence = await _old_execution_evidence(runtime.history, old_execution_id, runtime.default_principal)
        assert [(item.item_kind, item.content) for item in old.items if item.item_kind != "system"] == [
            ("user", "retained old question"), ("assistant", "done"),
        ]
        result = (await session.run("new question", timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert "retained old question" in observed_prompts
        assert "new question" in observed_prompts
        appended = await session.history()
        assert await _old_execution_evidence(runtime.history, old_execution_id, runtime.default_principal) == evidence
        assert appended.items[:len(old.items)] == old.items
        assert [(item.item_kind, item.content) for item in appended.items if item.item_kind != "system"] == [
            ("user", "retained old question"), ("assistant", "done"),
            ("user", "new question"), ("assistant", "done"),
        ]

    async with Runtime.open(
        "legacy-history", storage=RuntimeStorage.sqlite(path),
        models=helpers.RuntimeUsageModels(), capabilities=(application,),
    ) as reopened:
        restored = await reopened.agents.get("default").session("session").history()
        assert restored.items == appended.items
        assert await _old_execution_evidence(reopened.history, old_execution_id, reopened.default_principal) == evidence


@pytest.mark.asyncio
async def test_sqlite_explicit_checkpoint_boundary_excludes_later_observation(tmp_path: Path) -> None:
    from pydantic_ai.messages import TextPart

    from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint

    from .test_step_archive_read_boundaries import _archive

    async with _archive(tmp_path, "sqlite") as (_state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        first = ModelResponse(parts=[TextPart("checkpoint message")])
        later = ModelResponse(parts=[TextPart("later observation")])
        prepared = await archive.prepare_checkpoints(
            run, (AgentRunCheckpoint(
                "run", 1, [first], transcript_message_count_before=0,
            ),),
        )
        await archive.sync_prepared_projection(
            run, events=(), checkpoints=prepared.checkpoints,
            execution_id="execution", producer_generation=1,
        )
        observation = await archive.transcript_repository.prepare_observation(
            "run", (later,), first_message_index=1, pending=None, pending_keys=(),
        )
        await archive.sync_prepared_projection(
            run, events=(), checkpoints=(), observation=observation,
            execution_id="execution", producer_generation=1,
        )
        assert [message async for message in archive.iter_messages(agent_run_id="run")] == [first, later]
        checkpoint = await archive.latest_checkpoint(agent_run_id="run")
        assert checkpoint is not None
        assert checkpoint.messages == [first]


@pytest.mark.asyncio
async def test_sqlite_old_history_survives_snapshot_restore(tmp_path: Path) -> None:
    from linktools.ai.runtime.state import SnapshotLimits
    from linktools.ai.storage import InMemoryObjectStore

    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    sql = zlib.decompress(base64.b64decode(fixture["sql_zlib_base64"]))
    path = tmp_path / "history.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(sql.decode("utf-8"))
    principal = Principal("runtime", "default", "local_trusted")
    async with RuntimeHistory.open("legacy-history", storage=RuntimeStorage.sqlite(path)) as history:
        timeline = await history.session_timeline("session", principal=principal)
        execution_id = timeline.items[0].execution_id
        evidence = await _old_execution_evidence(history, execution_id, principal)
    state = RuntimeStorage.sqlite(path)
    await state.initialize(namespace="legacy-history", tenant_id="default", read_only=True)
    objects = InMemoryObjectStore("snapshot")
    limits = SnapshotLimits(max_entries=1000, max_bytes=1024 * 1024)
    try:
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    finally:
        await state.close()
    restored_root = tmp_path / "restored"
    await RuntimeStorage.restore_snapshot(
        snapshot, object_store=objects, root=restored_root, limits=limits,
    )
    application = CapabilityGroup("application")
    application.agent("default", model="default", allow_tools=())
    async with Runtime.open(
        "legacy-history", storage=RuntimeStorage.from_root(restored_root),
        models=helpers.RuntimeUsageModels(), capabilities=(application,),
    ) as runtime:
        assert await _old_execution_evidence(runtime.history, execution_id, runtime.default_principal) == evidence
        history = await runtime.agents.get("default").session("session").history()
        assert [(item.item_kind, item.content) for item in history.items if item.item_kind != "system"] == [
            ("user", "retained old question"), ("assistant", "done"),
        ]
