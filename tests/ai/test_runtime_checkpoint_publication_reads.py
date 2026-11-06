#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native recovery truth never replaces Trace publication or producer authority."""

from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.persistence._transcript_staging import StagedTranscript

from ._runtime_test_helpers import native_checkpoint, seed_native_snapshot
from .test_checkpoint_write_batches import _snapshot
from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def _producer(tmp_path: Path, backend: str, *, checkpointed: bool = True, unresolved: bool = False):
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="checkpoint-publication", tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="owner-claim", restart_checkpoints=checkpointed)
        await repository.create_execution(execution)
        store = state.run_store
        store.bind_step_store(native_checkpoint().store)
        store.bind_execution_producer(
            "run", execution_id="execution", producer_generation=1,
            conversation=False, restart_checkpoints=checkpointed, durable_trace=True,
        )
        await store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
        yield state, store, repository, execution
    finally:
        if unresolved:
            with pytest.raises(AIError) as error:
                await state.close()
            assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        else:
            await state.close()


def _checkpoint(count: int) -> AgentRunCheckpoint:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return AgentRunCheckpoint("run", count, [
        ModelRequest(parts=[UserPromptPart(str(index), timestamp=now)], run_id="run")
        for index in range(count)
    ], transcript_message_count_before=0, timestamp=now)


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_native_snapshot_and_trace_publication_use_independent_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    async with _producer(tmp_path, backend) as (_, store, _, _):
        checkpoint = _checkpoint(1)
        await seed_native_snapshot(store.step_store, run_id="run", messages=checkpoint.messages)
        source = await store.step_store.latest_snapshot(run_id="run")
        trace = store.read_store(PersistenceDomain.TRACE)

        async def forbidden(*args, **kwargs):
            raise AssertionError("Trace publication must not write native recovery snapshots")

        with monkeypatch.context() as patch:
            patch.setattr(store.step_store, "save_snapshot", forbidden)
            store.stage_transcript("run", StagedTranscript(tuple(checkpoint.messages)))
            await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
            assert await trace.latest_checkpoint(agent_run_id="run") is None
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
            before = await _snapshot(trace.state_store)
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert await _snapshot(trace.state_store) == before
        assert await trace.checkpoint_count(agent_run_id="run") == 1
        assert (await trace.latest_checkpoint(agent_run_id="run")).messages == checkpoint.messages
        assert await store.step_store.latest_snapshot(run_id="run") == source


@pytest.mark.parametrize("checkpointed", (False, True))
@pytest.mark.parametrize("condition", ("cancel", "takeover"))
async def test_trace_staging_does_not_grant_dispatch_or_replaced_producer_publication(
    tmp_path: Path, checkpointed: bool, condition: str,
) -> None:
    async with _producer(tmp_path, "sqlite", checkpointed=checkpointed) as (_, store, repository, execution):
        trace = store.read_store(PersistenceDomain.TRACE)
        before = await _snapshot(trace.state_store)
        await repository.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=1,
            next_record=replace(execution, revision=2,
                producer_generation=2 if condition == "takeover" else 1,
                producer_claim_id="replacement-claim" if condition == "takeover" else execution.producer_claim_id,
                status=(
                ExecutionStatus.CANCELLING if condition == "cancel" else ExecutionStatus.STARTED
            )),
        )
        checkpoint = _checkpoint(1)
        await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
        assert await _snapshot(trace.state_store) == before
        assert await store.step_store.latest_snapshot(run_id="run") is None
        with pytest.raises(AIError) as raised:
            await store.require_active_producer("run")
        assert raised.value.code is (ErrorCode.EXECUTION_CANCELLED if condition == "cancel" else ErrorCode.STORAGE_CONFLICT)
        if condition == "takeover":
            with pytest.raises(AIError) as rejected:
                await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
            assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
            assert await _snapshot(trace.state_store) == before
        else:
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
            assert (await trace.latest_checkpoint(agent_run_id="run")).messages == checkpoint.messages
            with pytest.raises(AIError) as stopped:
                await store.require_active_producer("run")
            assert stopped.value.code is ErrorCode.EXECUTION_CANCELLED
        assert await store.step_store.latest_snapshot(run_id="run") is None


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_native_checkpoint_readback_cannot_grant_current_producer_authority(
    tmp_path: Path, backend: str,
) -> None:
    from .test_model_control_observation import _advance_execution

    async with _producer(tmp_path, backend) as (_, store, repository, _):
        checkpoint = _checkpoint(1)
        await seed_native_snapshot(store.step_store, run_id="run", messages=checkpoint.messages)
        original = await store.read_store(PersistenceDomain.CHECKPOINT).latest_checkpoint(agent_run_id="run")
        await _advance_execution(repository)
        await store.checkpoint_persistence.materialize_recovery_checkpoint(agent_run_id="run", require_complete=True)
        assert await store.read_store(PersistenceDomain.CHECKPOINT).latest_checkpoint(agent_run_id="run") == original
        with pytest.raises(AIError) as raised:
            await store.require_active_producer("run")
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
        with pytest.raises(AIError) as rejected:
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        assert await store.read_store(PersistenceDomain.TRACE).latest_checkpoint(agent_run_id="run") is None
        assert await store.read_store(PersistenceDomain.CHECKPOINT).latest_checkpoint(agent_run_id="run") == original


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_trace_publication_rolls_back_execution_archive_and_content_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    async with _producer(tmp_path, backend) as (_, store, repository, _):
        archive = store.read_store(PersistenceDomain.TRACE)
        content = await archive.transcript_repository.content_archive(create=True)
        members = tuple(dict.fromkeys((repository.state_store, archive.state_store, content.store)))
        before = [await _snapshot(member) for member in members]
        checkpoint = _checkpoint(1)
        store.stage_transcript("run", StagedTranscript(tuple(checkpoint.messages)))
        await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
        publish = archive.sync_prepared_projection
        written = []

        async def rollback(*args, **kwargs):
            await publish(*args, **kwargs)
            written.append(True)
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with monkeypatch.context() as patch:
            patch.setattr(archive, "sync_prepared_projection", rollback)
            with pytest.raises(AIError) as raised:
                await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT and written == [True]
        assert [await _snapshot(member) for member in members] == before
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert (await archive.latest_checkpoint(agent_run_id="run")).messages == checkpoint.messages


@pytest.mark.parametrize("unknown", (False, True))
async def test_trace_lost_ack_uses_exact_fact_proof_or_keeps_unknown_fenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unknown: bool,
) -> None:
    async with _producer(tmp_path, "sqlite", unresolved=unknown) as (_, store, repository, _):
        archive = store.read_store(PersistenceDomain.TRACE)
        await archive.transcript_repository.content_archive(create=True)
        checkpoint = _checkpoint(1)
        store.stage_transcript("run", StagedTranscript(tuple(checkpoint.messages)))
        await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
        group = repository.state_store.storage_group
        mutate = group.mutate
        verify = archive.verify_checkpoint_range
        commits, proofs = [], []
        depth = 0

        async def lost_reply(stores, callback):
            nonlocal depth
            if depth:
                return await mutate(stores, callback)
            depth += 1
            try:
                result = await mutate(stores, callback)
            finally:
                depth -= 1
            commits.append(result)
            raise OSError("Trace commit acknowledgement lost")

        async def readback(**kwargs):
            proofs.append(kwargs["after_sequence"])
            if unknown:
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return await verify(**kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(group, "mutate", lost_reply)
            patch.setattr(archive, "verify_checkpoint_range", readback)
            if unknown:
                with pytest.raises(AIError) as raised:
                    await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
                assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
                with pytest.raises(AIError) as fenced:
                    await store.save_checkpoint(_checkpoint(2), execution_id="execution", producer_generation=1)
                assert fenced.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            else:
                await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert len(commits) == 1 and proofs == [0]
        assert await archive.checkpoint_count(agent_run_id="run") == 1
        assert (await archive.latest_checkpoint(agent_run_id="run")).messages == checkpoint.messages
        assert await store.step_store.latest_snapshot(run_id="run") is None
        if not unknown:
            before = await _snapshot(archive.state_store)
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
            assert await _snapshot(archive.state_store) == before


def _stage_request(store, run_id: str, sequence: int, messages) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart

    from linktools.ai.core import UsageMetrics
    from linktools.ai.persistence._model_interaction import StagedModelInteraction, build_inline_context_projection

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    intern = lambda payload: store.intern_payload(run_id, payload)
    store.stage_model_interaction(StagedModelInteraction(
        agent_run_id=run_id, step_index=sequence, model_request_seq=sequence,
        purpose="agent", output_retry_index=None, model={"route_id": "default"},
        request_context=build_inline_context_projection(messages, intern),
        request_envelope_digest=intern(b"{}")[0],
        response_context=build_inline_context_projection((
            ModelResponse(parts=[TextPart(f"response-{sequence}")], timestamp=now, run_id=run_id),
        ), intern),
        status="SUCCEEDED", error_code=None, duration_ns=1, usage=UsageMetrics(model_requests=1),
        started_at=now, finished_at=now,
    ))


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("outcome", ("rollback", "unknown", "lost_ack"))
async def test_request_projection_advances_only_after_confirmed_trace_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, outcome: str,
) -> None:
    async with _producer(tmp_path, backend, unresolved=outcome == "unknown") as (_, store, repository, _):
        archive = store.read_store(PersistenceDomain.TRACE)
        _stage_request(store, "run", 1, _checkpoint(1).messages)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        offset = store.trace_persistence.offsets["run"]
        baseline = offset.request_projection
        assert baseline is not None
        original = (await archive.list_model_interactions(agent_run_id="run"))[0]
        assert baseline == original.request_context
        _stage_request(store, "run", 2, _checkpoint(2).messages)
        prepare = archive.prepare_interactions
        publish = archive.sync_prepared_projection
        group = repository.state_store.storage_group
        mutate = group.mutate
        proofs = archive.verify_checkpoint_range
        candidates, received = [], []
        depth = 0

        async def prepared(*args, **kwargs):
            received.append(kwargs.get("baseline"))
            result = await prepare(*args, **kwargs)
            candidates.append(result[-1].request_context)
            return result

        async def rollback(*args, **kwargs):
            await publish(*args, **kwargs)
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        async def lose_ack(stores, callback):
            nonlocal depth
            if depth:
                return await mutate(stores, callback)
            depth += 1
            try:
                await mutate(stores, callback)
            finally:
                depth -= 1
            raise OSError("request projection acknowledgement lost")

        async def unknown_proof(**kwargs):
            if outcome == "unknown":
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return await proofs(**kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(archive, "prepare_interactions", prepared)
            if outcome == "rollback":
                patch.setattr(archive, "sync_prepared_projection", rollback)
            else:
                patch.setattr(group, "mutate", lose_ack)
                patch.setattr(archive, "verify_checkpoint_range", unknown_proof)
            if outcome == "lost_ack":
                await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
            else:
                with pytest.raises(AIError) as caught:
                    await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
                assert caught.value.code is (
                    ErrorCode.STORAGE_CONFLICT if outcome == "rollback" else ErrorCode.STORAGE_COMMIT_UNKNOWN
                )
        assert received == [baseline] and len(candidates) == 1
        assert candidates[0] is not None and candidates[0] != baseline
        if outcome == "lost_ack":
            assert offset.request_projection == candidates[0]
        else:
            assert offset.request_projection is baseline
        archived = await archive.list_model_interactions(agent_run_id="run")
        assert len(archived) == (1 if outcome == "rollback" else 2)
        assert archived[0] == original
        if outcome == "rollback":
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
            assert offset.request_projection == candidates[0]
        elif outcome == "unknown":
            with pytest.raises(AIError) as fenced:
                await store.capture_execution_projection("run")
            assert fenced.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            assert offset.request_projection is baseline


@pytest.mark.parametrize("variant", ("equal", "append", "same_length", "shorter", "reordered", "summary", "tool"))
async def test_confirmed_request_projection_rewrites_exact_changed_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str,
) -> None:
    from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart

    async with _producer(tmp_path, "sqlite") as (_, store, _, _):
        archive = store.read_store(PersistenceDomain.TRACE)
        initial = _checkpoint(2).messages
        _stage_request(store, "run", 1, initial)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        original = (await archive.list_model_interactions(agent_run_id="run"))[0]
        baseline = store.trace_persistence.offsets["run"].request_projection
        messages = _checkpoint(3 if variant == "append" else 2).messages
        if variant == "same_length":
            messages[0].parts[0].content = "same-length rewrite"
        elif variant == "shorter":
            messages = messages[:1]
        elif variant == "reordered":
            messages.reverse()
        elif variant == "summary":
            messages = [ModelRequest(parts=[UserPromptPart("summary")], run_id="run")]
        elif variant == "tool":
            messages = [
                ModelResponse(parts=[ToolCallPart("lookup", {}, "call")], run_id="run"),
                ModelRequest(parts=[ToolReturnPart("lookup", "result", "call")], run_id="run"),
            ]
        seen = []
        prepare = archive.prepare_interactions

        async def observed(*args, **kwargs):
            seen.append(kwargs.get("baseline"))
            return await prepare(*args, **kwargs)

        _stage_request(store, "run", 2, messages)
        with monkeypatch.context() as patch:
            patch.setattr(archive, "prepare_interactions", observed)
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert seen == [baseline]
        archived = await archive.list_model_interactions(agent_run_id="run")
        resolved = await archive.resolve_model_interactions(archived)
        assert list(resolved[0][0]) == initial
        assert list(resolved[1][0]) == messages
        assert archived[0] == original
        assert store.trace_persistence.offsets["run"].request_projection == archived[-1].request_context
        if variant == "equal":
            assert archived[-1].request_context == baseline
        else:
            assert archived[-1].request_context != baseline


async def test_request_projection_cache_is_run_local_and_released_with_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _producer(tmp_path, "memory") as (_, store, _, _):
        archive = store.read_store(PersistenceDomain.TRACE)
        _stage_request(store, "run", 1, _checkpoint(1).messages)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        baseline = store.trace_persistence.offsets["run"].request_projection
        store.bind_execution_producer("other", execution_id="execution", producer_generation=1)
        await store.register_agent_run(AgentRunRecord("other"), execution_id="execution")
        other_messages = [ModelRequest(parts=[UserPromptPart("other run")], run_id="other")]
        _stage_request(store, "other", 1, other_messages)
        prepare = archive.prepare_interactions
        received = []

        async def observed(run, *args, **kwargs):
            received.append((run.agent_run_id, kwargs.get("baseline")))
            return await prepare(run, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(archive, "prepare_interactions", observed)
            await store.flush_execution_projection("other", execution_id="execution", producer_generation=1)
        assert received == [("other", None)]
        assert store.trace_persistence.offsets["run"].request_projection is baseline
        other = store.trace_persistence.offsets["other"].request_projection
        assert other is not None and other != baseline
        await store.release_staging_many(candidate_agent_run_ids=("run",), execution_id="execution")
        assert "run" not in store.trace_persistence.offsets
        assert store.trace_persistence.offsets["other"].request_projection is other


async def test_cold_trace_reader_rebuilds_without_persisted_request_projection_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _producer(tmp_path, "sqlite", checkpointed=False) as (_, store, _, _):
        run = await store.get_agent_run(agent_run_id="run")
        _stage_request(store, "run", 1, _checkpoint(1).messages)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        archive = store.read_store(PersistenceDomain.TRACE)
        before = await _snapshot(archive.state_store)
        kinds = {record.kind for record in before[0]}
        assert store.trace_persistence.offsets["run"].request_projection is not None
    reopened = Persistence(_storage("sqlite", tmp_path))
    await reopened.initialize(namespace="checkpoint-publication", tenant_id="tenant")
    try:
        store = reopened.run_store
        store.bind_execution_producer(
            "run", execution_id="execution", producer_generation=1,
            conversation=False, restart_checkpoints=False, durable_trace=True,
        )
        await store.register_agent_run(run, execution_id="execution")
        assert store.trace_persistence.offsets["run"].request_projection is None
        archive = store.read_store(PersistenceDomain.TRACE)
        prepare = archive.prepare_interactions
        received = []

        async def observed(*args, **kwargs):
            received.append(kwargs.get("baseline"))
            return await prepare(*args, **kwargs)

        _stage_request(store, "run", 2, _checkpoint(2).messages)
        with monkeypatch.context() as patch:
            patch.setattr(archive, "prepare_interactions", observed)
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert received == [None]
        archived = await archive.list_model_interactions(agent_run_id="run")
        assert store.trace_persistence.offsets["run"].request_projection == archived[-1].request_context
        resolved = await archive.resolve_model_interactions(archived)
        assert [len(value[0]) for value in resolved] == [1, 2]
        assert {record.kind for record in (await _snapshot(archive.state_store))[0]} == kinds
    finally:
        await reopened.close()


async def test_cached_request_projection_does_not_bypass_changed_producer_authority(tmp_path: Path) -> None:
    from .test_model_control_observation import _advance_execution

    async with _producer(tmp_path, "sqlite") as (_, store, repository, _):
        archive = store.read_store(PersistenceDomain.TRACE)
        _stage_request(store, "run", 1, _checkpoint(1).messages)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        baseline = store.trace_persistence.offsets["run"].request_projection
        before = await _snapshot(archive.state_store)
        await _advance_execution(repository)
        _stage_request(store, "run", 2, _checkpoint(2).messages)
        with pytest.raises(AIError) as denied:
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert denied.value.code is ErrorCode.STORAGE_CONFLICT
        assert store.trace_persistence.offsets["run"].request_projection is baseline
        assert await _snapshot(archive.state_store) == before
        await store.discard_revoked_producer_staging("run", execution_id="execution", producer_generation=1)
