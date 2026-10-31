#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint handoff retains admitted terminal Trace data across Runtime owners."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import CheckpointCapability, ExecutionCapability, TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._projection_payload import decode_projection_payload
from linktools.ai.runtime import Runtime
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import model_features
from .test_explicit_runtime_features import _FeatureFunctionModel, _FeatureModels, _features


@pytest.mark.asyncio
async def test_terminal_trace_handoff_recovers_same_execution_without_losing_admitted_facts(tmp_path: Path) -> None:
    entered = asyncio.Event()
    calls = []

    class Buffer(InMemoryCacheBackend):
        close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            await super().close()

    buffer = Buffer(max_bytes=4 * 1024 * 1024)

    async def blocked(messages, info):
        calls.append("owner")
        entered.set()
        await asyncio.Event().wait()
        return ModelResponse(parts=[TextPart("unreachable")])

    async def answer(messages, info):
        calls.append("successor")
        return ModelResponse(parts=[TextPart("recovered answer")])

    def features(model):
        return model_features(_FeatureModels(_FeatureFunctionModel(model)), _features(
            ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()),
            TraceCapability(commit=CommitPolicy.terminal(buffer=buffer)),
            CheckpointCapability(),
        ))

    database = tmp_path / "state.db"
    async with Persistence.open("terminal-trace-handoff", storage=PersistenceStorage.sqlite(database)) as state:
        async with Runtime.attach(persistence=state, features=features(blocked), metrics=None) as owner:
            session = await owner.agents.get().create_session("session")
            execution = await session.start("Remember blue", idempotency_key="turn")
            await asyncio.wait_for(entered.wait(), 5)
            scopes = tuple(value.id for value in owner._run_store.trace_persistence._buffers.values())
            entries = [entry for scope in scopes for entry in await buffer.read_batch(scope, limit=100)]
            assert entries and all(isinstance(entry[2], bytes) for entry in entries)
            captured = [decode_projection_payload(entry[2]) for entry in entries]
            admitted_events = {capture.run.agent_run_id: capture.events for capture in captured}
            assert any(admitted_events.values())
            archive = owner._run_store.read_store(PersistenceDomain.TRACE)
            unpublished = []
            for run_id, events in admitted_events.items():
                source_events = await archive.list_events(agent_run_id=run_id)
                unpublished.extend(event for event in events if event not in source_events)
            assert unpublished
            await asyncio.wait_for(owner.handoff(), 10)
            await owner.close()
            assert calls == ["owner"]
            for scope in scopes:
                assert await buffer.read_batch(scope, limit=100) == ()
            assert buffer.close_calls == 0

    async with Persistence.open("terminal-trace-handoff", storage=PersistenceStorage.sqlite(database)) as state:
        async with Runtime.attach(persistence=state, features=features(answer), metrics=None,
                                  auto_recover=False) as successor:
            principal = successor.default_principal
            before = await successor.executions.inspect(execution.execution_id, principal=principal)
            assert before.status is ExecutionStatus.STARTED
            archive = successor._run_store.read_store(PersistenceDomain.TRACE)
            for run_id, events in admitted_events.items():
                source_events = await archive.list_events(agent_run_id=run_id)
                assert all(event in source_events for event in events)
                interactions = await archive.list_model_interactions(agent_run_id=run_id)
                assert interactions
                resolved = await archive.resolve_model_interactions(interactions)
                assert "Remember blue" in repr(resolved)
            recovered = await successor.executions.recover(execution.execution_id, principal=principal,
                                                           idempotency_key="recover-turn")
            assert recovered.execution_id == execution.execution_id
            await successor.executions.wait(execution.execution_id, principal=principal, timeout_seconds=10)
            assert (await successor.executions.inspect(execution.execution_id,
                                                      principal=principal)).status is ExecutionStatus.SUCCEEDED
            archive = successor._run_store.read_store(PersistenceDomain.TRACE)
            for run_id, events in admitted_events.items():
                source_events = await archive.list_events(agent_run_id=run_id)
                assert all(event in source_events for event in events)
            assert calls == ["owner", "successor"]
            await successor.close()
    assert buffer.close_calls == 0
