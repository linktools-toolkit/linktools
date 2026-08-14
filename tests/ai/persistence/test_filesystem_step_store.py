#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Crash-safe filesystem StepStore conformance checks."""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.adapter import FilesystemStepArchive
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)


def _run(run_id: str) -> RunRecord:
    return RunRecord(run_id=run_id, conversation_id="c-conversation", parent_run_id=None, agent_name="agent", metadata={"segment_sequence": "1", "agent_name": "agent"}, started_at=datetime.now(timezone.utc))


def _archive(root: Path, runtime_domain: RuntimeDomain) -> FilesystemStepArchive:
    return FilesystemStepArchive(root, namespace="namespace", tenant_id="tenant", runtime_domain=runtime_domain)


@pytest.mark.asyncio
async def test_filesystem_step_store_round_trip_and_restart(tmp_path: Path) -> None:
    first = _archive(tmp_path, RuntimeDomain.EXECUTION)
    await first.initialize()
    run = _run("r-run")
    await first.register_run(run)
    await first.append_event(StepEvent(run_id=run.run_id, kind="model_request_started", step_index=1, timestamp=datetime.now(timezone.utc), conversation_id=run.conversation_id, parent_run_id=None, agent_name="agent", tool_call_id=None, tool_name=None, error=None, metadata={}))
    await first.save_snapshot(ContinuableSnapshot(run_id=run.run_id, step_index=1, messages=[ModelRequest(parts=[UserPromptPart(content="hello")], conversation_id=run.conversation_id)], conversation_id=run.conversation_id, parent_run_id=None, agent_name="agent", timestamp=datetime.now(timezone.utc)))
    await first.register_run(run)
    await first.close()

    second = _archive(tmp_path, RuntimeDomain.EXECUTION)
    await second.initialize()
    assert await second.get_run(run_id=run.run_id) == run
    assert len(await second.list_events(run_id=run.run_id)) == 1
    assert await second.latest_snapshot(run_id=run.run_id) is not None
    await second.close()
    recovery = _archive(tmp_path, RuntimeDomain.RECOVERY)
    await recovery.initialize()
    await recovery.register_run(run)
    await recovery.record_tool_effect(ToolEffectRecord(tool_call_id="call", tool_name="tool", run_id=run.run_id, status="started", started_at=datetime.now(timezone.utc), ended_at=None, idempotency_key="key", effect_summary=None))
    assert len(await recovery.list_unresolved_tool_effects(run_id=run.run_id)) == 1
    await recovery.close()


@pytest.mark.asyncio
async def test_filesystem_step_store_persists_reachable_media_before_snapshot(tmp_path: Path) -> None:
    first = _archive(tmp_path, RuntimeDomain.EXECUTION)
    await first.initialize()
    run = _run("media-run")
    await first.register_run(run)
    data = b"media" * 16000
    message = ModelRequest(
        parts=[UserPromptPart(content=[BinaryContent(data, media_type="application/octet-stream")])],
        conversation_id=run.conversation_id,
    )
    snapshot = ContinuableSnapshot(
        run_id=run.run_id,
        step_index=1,
        messages=[message],
        conversation_id=run.conversation_id,
        parent_run_id=None,
        agent_name="agent",
        timestamp=datetime.now(timezone.utc),
    )
    await first.save_snapshot(snapshot)
    await first.close()

    assert any(path.is_file() and path.name == hashlib.sha256(data).hexdigest() for path in tmp_path.rglob("*"))
    second = _archive(tmp_path, RuntimeDomain.EXECUTION)
    await second.initialize()
    restored = await second.latest_snapshot(run_id=run.run_id)
    assert restored is not None
    assert restored.messages[0].parts[0].content[0].data == data
    await second.close()


@pytest.mark.asyncio
async def test_filesystem_step_store_rejects_path_identifiers(tmp_path: Path) -> None:
    store = _archive(tmp_path, RuntimeDomain.EXECUTION)
    await store.initialize()
    with pytest.raises(AIError) as error:
        await store.register_run(_run("../escape"))
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    await store.close()


@pytest.mark.asyncio
async def test_filesystem_step_store_hides_interrupted_snapshots_and_rejects_unknown_state(tmp_path: Path) -> None:
    store = _archive(tmp_path, RuntimeDomain.EXECUTION)
    await store.initialize()
    run = _run("r-run")
    await store.register_run(run)
    message = ModelRequest(parts=[UserPromptPart(content="hello")], conversation_id=run.conversation_id)
    complete = ContinuableSnapshot(run_id=run.run_id, step_index=1, messages=[message], conversation_id=run.conversation_id, parent_run_id=None, agent_name="agent", timestamp=datetime.now(timezone.utc), state="complete")
    interrupted = ContinuableSnapshot(run_id=run.run_id, step_index=2, messages=[message], conversation_id=run.conversation_id, parent_run_id=None, agent_name="agent", timestamp=datetime.now(timezone.utc), state="interrupted")
    await store.save_snapshot(complete)
    await store.save_snapshot(interrupted)
    assert await store.latest_snapshot(run_id=run.run_id) is None
    assert await store.latest_snapshot(run_id=run.run_id, include_interrupted=True) == interrupted
    snapshot_path = sorted(tmp_path.rglob("snapshots/snapshot-*.json"))[-1]
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["state"] = "unknown"
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

    await store.close()
    reopened = _archive(tmp_path, RuntimeDomain.EXECUTION)
    with pytest.raises(AIError) as error:
        await reopened.initialize()
    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED


def test_filesystem_step_store_writes_hashed_run_directory(tmp_path: Path) -> None:
    async def run() -> Path:
        store = _archive(tmp_path, RuntimeDomain.EXECUTION)
        await store.initialize()
        await store.register_run(_run("r-run"))
        path = next(tmp_path.rglob("run.json"))
        await store.close()
        return path

    path = asyncio.run(run())
    assert path.parent.name != "r-run"
    assert (path.parent / "run.json").is_file()
