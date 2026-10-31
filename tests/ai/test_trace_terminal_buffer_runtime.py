#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Trace terminal(buffer) uses bytes and preserves source control boundaries."""

import asyncio

import pytest
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.runtime import Runtime
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import model_features
from .test_explicit_runtime_features import _FeatureFunctionModel, _FeatureModels, _features
from .test_trace_buffer_runtime_failure import _track_source_transactions


@pytest.mark.asyncio
@pytest.mark.parametrize("retain_history", [False, True])
@pytest.mark.parametrize("provider_failure", [False, True])
async def test_public_terminal_trace_payload_and_control_share_transaction(
    tmp_path, monkeypatch, retain_history, provider_failure,
):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        if provider_failure:
            raise ModelAPIError("test", "provider failed")
        return ModelResponse(parts=[TextPart("buffered response")])

    buffer = InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)
    features = model_features(_FeatureModels(_FeatureFunctionModel(model)), _features(
        ExecutionCapability(retain_history=retain_history, commit=CommitPolicy.sync()),
        TraceCapability(commit=CommitPolicy.terminal(buffer=buffer)),
    ))
    async with Persistence.open("public-terminal-trace", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence:
        async with Runtime.attach(persistence=persistence, features=features, metrics=None) as runtime:
            store = runtime._run_store
            archive = store.read_store(PersistenceDomain.TRACE)
            repository = persistence.execution.executions
            trace_frames, terminal_frames = _track_source_transactions(monkeypatch, archive, repository)
            execution = await runtime.agents.get().start("question")
            try:
                await asyncio.wait_for(entered.wait(), 5)
                queued = tuple(store.trace_persistence._buffers.values())
                assert len(queued) == 1
                entries = await buffer.read_batch(queued[0].id, limit=100)
                assert entries and all(isinstance(value[2], bytes) and value[2] for value in entries)
                assert await archive.list_agent_runs() == [] and not trace_frames
                current = await repository.get(execution.execution_id, tenant_id=runtime.tenant_id)
                assert current.status is ExecutionStatus.STARTED and current.conversation == retain_history
            finally:
                release.set()
            result = (await execution.wait(timeout_seconds=5)).result
            assert result.status is (ExecutionStatus.FAILED if provider_failure else ExecutionStatus.SUCCEEDED)
            assert calls == 1 and len(terminal_frames) == 1
            assert trace_frames and all(value is terminal_frames[0] for value in trace_frames)
            runs = await archive.list_agent_runs()
            assert len(runs) == 1
            interactions = await archive.list_model_interactions(agent_run_id=runs[0].agent_run_id)
            assert len(interactions) == 1
            assert interactions[0].status == ("FAILED" if provider_failure else "SUCCEEDED")
            await store.preflight_close()
            assert await buffer.read_batch(queued[0].id, limit=100) == ()



@pytest.mark.asyncio
async def test_public_terminal_trace_scopes_are_isolated_between_runtimes(tmp_path):
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    buffer = InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)

    def features(index):
        async def model(messages, info):
            entered[index].set()
            await release[index].wait()
            return ModelResponse(parts=[TextPart(f"response {index}")])
        return model_features(_FeatureModels(_FeatureFunctionModel(model)), _features(
            TraceCapability(commit=CommitPolicy.terminal(buffer=buffer)),
        ))

    async with (
        Persistence.open("trace-scope-owner", storage=PersistenceStorage.sqlite(tmp_path / "first.db")) as first_state,
        Persistence.open("trace-scope-owner", storage=PersistenceStorage.sqlite(tmp_path / "second.db")) as second_state,
        Runtime.attach(persistence=first_state, features=features(0), metrics=None) as first,
        Runtime.attach(persistence=second_state, features=features(1), metrics=None) as second,
    ):
        first_execution = await first.agents.get().start("first question")
        second_execution = await second.agents.get().start("second question")
        try:
            await asyncio.wait_for(asyncio.gather(*(value.wait() for value in entered)), 5)
            first_scope = next(iter(first._run_store.trace_persistence._buffers.values())).id
            second_scope = next(iter(second._run_store.trace_persistence._buffers.values())).id
            assert first_scope != second_scope
            second_entries = await buffer.read_batch(second_scope, limit=100)
            assert second_entries
            release[0].set()
            assert (await first_execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
            await first.close()
            assert await buffer.read_batch(first_scope, limit=100) == ()
            assert await buffer.read_batch(second_scope, limit=100) == second_entries
            second_archive = second._run_store.read_store(PersistenceDomain.TRACE)
            assert await second_archive.list_agent_runs() == []
            release[1].set()
            assert (await second_execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
            await second._run_store.preflight_close()
            assert await buffer.read_batch(second_scope, limit=100) == ()
            assert len(await second_archive.list_agent_runs()) == 1
        finally:
            for value in release:
                value.set()
