#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable native terminal results remain live until their last owner retires."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import text as sqltext

from linktools.ai.core import ExecutionStatus, ExecutionEventType, StopReason, UsageMetrics
from linktools.ai.persistence import Persistence, PersistenceStorage, PersistenceDomain, SqlStepStore
from linktools.ai.persistence._archive_closure import iter_native_message_roots
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._step_contracts import AgentRunRecord, AgentRunCheckpoint
from linktools.ai.persistence.codec import decode_envelope, execution_payload_slots, native_text_reference
from linktools.ai.persistence.contracts import ExecutionTerminalCommit, ResultRecord
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.storage import StoredPayload, InMemoryObjectStore

from .test_step_projection import _execution


class Guard:
    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        yield


@pytest.mark.asyncio
async def test_native_result_readonly_snapshot_and_last_owner_gc(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    text = "terminal body " * 10000
    objects = InMemoryObjectStore("native-result-portable")
    limits = SnapshotLimits(10000, 20_000_000)
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.sqlite(path)) as state:
        repo = state.execution.executions
        store = repo.state_store
        native = SqlStepStore.for_state_store(store, namespace="native-result", tenant_id="tenant")
        state.run_store.bind_step_store(native)
        current = replace(_execution(), revision=1, producer_generation=1)
        await repo.create_execution(current)
        run = AgentRunRecord("run")
        checkpoint = AgentRunCheckpoint("run", 1, [ModelResponse(parts=[TextPart(text)])],
            transcript_message_count_before=0, timestamp=current.updated_at)
        commands = StateCommands(repo, namespace="native-result", events=state.execution.events,
            execution_run_store=state.run_store.read_store(PersistenceDomain.TRACE), background_tasks=set())
        output = StoredPayload.inline_json(text)
        result = ResultRecord(output, StopReason.END_TURN, UsageMetrics(), current.updated_at)
        commit = ExecutionTerminalCommit(current.revision, current.event_seq,
            replace(current, status=ExecutionStatus.SUCCEEDED), result, ExecutionEventType.EXECUTION_SUCCEEDED,
            {"agent_run_id": "run"}, producer_generation=1)
        await commands.commit_terminal_checkpoint(commit, expected_execution=current,
            execution_run=run, execution_checkpoints=(checkpoint,))
        raw = await store.read_record(repo.execution_record_key("execution"))
        assert any(native_text_reference(payload) for _, payload in
                   execution_payload_slots(decode_envelope(raw.data).value["payload"]))
        loaded = await repo.get_result_record("execution", tenant_id="tenant", fresh=True)
        assert loaded.result.output == output
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.sqlite(path),
                                read_only=True) as reader:
        snapshot = await reader.export_snapshot(object_store=objects, limits=limits)
    target = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=target, limits=limits)
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.from_root(target)) as restored:
        repo = restored.execution.executions
        result = await repo.get_result_record("execution", tenant_id="tenant", fresh=True)
        assert result.result.output == output
        store = repo.state_store
        async with store.storage_group.context.engine.connect() as connection:
            before = await connection.scalar(sqltext("select count(*) from ai_step_messages"))
        assert before > 0
        # Remove transcript owners so the compact result alone protects its body.
        for partition in await store.storage_group.scan_stores():
            rows = await partition.read(lambda tx: tx.scan_records_page(after=None, limit=1000))
            keys = tuple(row.key for row in rows if row.kind != "execution"
                         and tuple(iter_native_message_roots(row.data)))
            if keys:
                await partition.mutate(lambda tx: tx.delete_records(keys))
            facts = await partition.read(lambda tx: tx.scan_facts_page(after=None, limit=1000))
            owners = {fact.owner_key for fact in facts if tuple(iter_native_message_roots(fact.data))}
            for owner in owners:
                await partition.mutate(lambda tx: tx.delete_fact_streams(owner))
        await restored.purge_unreferenced_objects((), exclusive=Guard())
        async with store.storage_group.context.engine.connect() as connection:
            retained = await connection.scalar(sqltext("select count(*) from ai_step_messages"))
        assert retained > 0
        assert (await repo.get_result_record("execution", tenant_id="tenant", fresh=True)).result.output == output
        await store.mutate(lambda tx: tx.delete_records((repo.execution_record_key("execution"),)))
        await restored.purge_unreferenced_objects((), exclusive=Guard())
        async with store.storage_group.context.engine.connect() as connection:
            after = await connection.scalar(sqltext("select count(*) from ai_step_messages"))
            nodes = await connection.scalar(sqltext("select count(*) from ai_step_sequences"))
        assert (after, nodes) == (0, 0)
