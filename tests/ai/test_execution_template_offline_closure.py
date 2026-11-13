#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Whole terminal templates keep Native leaves valid through offline retention."""

import pytest
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from dataclasses import replace
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pydantic_ai.messages import ModelResponse,TextPart
from linktools.ai.persistence import Persistence,PersistenceStorage,PersistenceDomain,SqlStepStore
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._step_contracts import AgentRunRecord,AgentRunCheckpoint
from linktools.ai.persistence.contracts import ExecutionTerminalCommit,ResultRecord
from linktools.ai.core import ExecutionStatus,ExecutionEventType,StopReason,UsageMetrics,canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import StoredPayload,InMemoryObjectStore,read_object
from linktools.ai.persistence._snapshot import SnapshotLimits
from .test_step_projection import _execution
class Guard:
    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("native_result", (True, False))
async def test_terminal_template_gc_snapshot_roundtrip(tmp_path: Path, native_result: bool) -> None:
    path=tmp_path/'db';storage=PersistenceStorage.sqlite(path)
    async with Persistence.open('native-result',tenant_id='tenant',storage=storage) as state:
        repo=state.execution.executions;store=repo.state_store
        native=SqlStepStore.for_state_store(store,namespace='native-result',tenant_id='tenant');state.run_store.bind_step_store(native)
        current=replace(_execution(),revision=1,producer_generation=1);await repo.create_execution(current)
        run=AgentRunRecord('run');checkpoint=AgentRunCheckpoint('run',1,[ModelResponse(parts=[TextPart('done')])],transcript_message_count_before=0,timestamp=current.updated_at)
        commands=StateCommands(repo,namespace='native-result',events=state.execution.events,execution_run_store=state.run_store.read_store(PersistenceDomain.TRACE),background_tasks=set())
        output=StoredPayload.inline_json('done' if native_result else {'structured':'done'})
        result=ResultRecord(output,StopReason.END_TURN,UsageMetrics(),current.updated_at)
        commit=ExecutionTerminalCommit(current.revision,current.event_seq,replace(current,status=ExecutionStatus.SUCCEEDED),result,ExecutionEventType.EXECUTION_SUCCEEDED,{'agent_run_id':'run'},producer_generation=1)
        await commands.commit_terminal_checkpoint(commit,expected_execution=current,execution_run=run,execution_checkpoints=(checkpoint,))
        await state.purge_unreferenced_objects((),exclusive=Guard())
        assert (await repo.get_result_record('execution',tenant_id='tenant',fresh=True)).result==result
        assert (await repo.get_fresh('execution',tenant_id='tenant')).result==result
        objects=InMemoryObjectStore();limits=SnapshotLimits(10000,20_000_000)
    async with Persistence.open('native-result',tenant_id='tenant',storage=storage,read_only=True) as reader:
        snapshot=await reader.export_snapshot(object_store=objects,limits=limits)
    manifest = json.loads(await read_object(objects, snapshot.key,
        expected_digest=snapshot.digest, expected_size=snapshot.size))
    assert any(domain["facts"] and domain["sequences"] for domain in manifest["domains"].values())
    target=tmp_path/'restored';await Persistence.restore_snapshot(snapshot,object_store=objects,root=target,limits=limits)
    async with Persistence.open('native-result',tenant_id='tenant',storage=PersistenceStorage.from_root(target)) as restored:
        assert (await restored.execution.executions.get_result_record('execution',tenant_id='tenant')).result==result
        await restored.purge_unreferenced_objects((),exclusive=Guard())
    # Facts/sequences must survive decoding without bypassing the retained
    # content closure. A transport with a missing live Native body is invalid.
    damaged = deepcopy(manifest)
    assert damaged["native_messages"][0]["messages"]
    damaged["native_messages"][0]["messages"].clear()
    raw = canonical_json_bytes(damaged)
    bad_ref = replace(snapshot, key="damaged-native-snapshot", digest=hashlib.sha256(raw).hexdigest(), size=len(raw))

    async def chunks() -> AsyncIterator[bytes]:
        yield raw

    await objects.put(bad_ref.key, chunks(), expected_size=bad_ref.size, expected_digest=bad_ref.digest)
    with pytest.raises(AIError) as failure:
        await Persistence.restore_snapshot(bad_ref, object_store=objects, root=tmp_path / "rejected", limits=limits)
    assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
