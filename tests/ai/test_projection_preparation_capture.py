#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Projection source capture preserves preparation and releases its read scope."""
from pathlib import Path
from dataclasses import replace

import pytest

from linktools.ai.persistence import Persistence
from linktools.ai.persistence._step_archive import CapturedExecutionProjection
from linktools.ai.persistence._step_contracts import AgentRunRecord
from linktools.ai.persistence.state import active_state_scope

from .test_checkpoint_metadata_snapshot_io import (
    _assert_prepared_values_equal, _checkpoint, _trace_archive,
)
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("checkpoint", [False, True])
async def test_projection_capture_matches_preparation_without_holding_source_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, checkpoint: bool,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="projection-capture", tenant_id="tenant")
    try:
        archive = await _trace_archive(state)
        run = AgentRunRecord("run")
        checkpoints = (_checkpoint(1),) if checkpoint else ()
        expected = await archive.prepare_checkpoints(run, checkpoints)
        prepare = archive.transcript_repository.prepare_model_projection

        async def outside_scope(*args, **kwargs):
            assert active_state_scope() is None
            return await prepare(*args, **kwargs)

        monkeypatch.setattr(archive.transcript_repository, "prepare_model_projection", outside_scope)
        captured = CapturedExecutionProjection(run, (), checkpoints, 0, 0, 0, len(checkpoints))
        result = await archive.prepare_captured_projection(captured, baseline=None)
        assert result is not None
        prepared, observation, interactions, bundle, head = result
        assert observation is None and interactions == ()
        assert (head.event_count, head.checkpoint_count, head.transcript_message_count) == (0, 0, 0)
        prepared = replace(prepared, checkpoints=tuple(
            replace(value, archive=bundle) for value in prepared.checkpoints
        ))
        await _assert_prepared_values_equal(archive, expected, prepared)
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_captured_observation_keeps_stale_publication_fence(
    tmp_path: Path, backend: str,
) -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence._transcript_staging import StagedTranscript

    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="projection-fence", tenant_id="tenant")
    try:
        archive = await _trace_archive(state)
        run = AgentRunRecord("run")
        await archive.materialize_checkpoint(run, _checkpoint(0), execution_id="execution")
        message = ModelRequest(parts=[UserPromptPart("one")])
        captured = CapturedExecutionProjection(
            run, (), (), 0, 1, 0, 1, transcript=StagedTranscript((message,)),
            target_message_index=1,
        )
        result = await archive.prepare_captured_projection(captured, baseline=None)
        assert result is not None and result[1] is not None
        first = result[1]
        other = await archive.transcript_repository.prepare_observation(
            "run", (message,), first_message_index=0, pending=None, pending_keys=(),
        )
        await archive.state_store.mutate(
            lambda tx: archive.transcript_repository.commit_observation(tx, other),
        )
        with pytest.raises(AIError) as stale:
            await archive.state_store.mutate(
                lambda tx: archive.transcript_repository.commit_observation(tx, first),
            )
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
    finally:
        await state.close()


async def test_projection_preparation_groups_only_source_metadata(tmp_path: Path) -> None:
    from sqlalchemy import event
    from linktools.ai.persistence._archive import PreparedArchive
    from linktools.ai.persistence._transcript_staging import StagedTranscript

    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="projection-reads", tenant_id="tenant")
    try:
        archive = await _trace_archive(state)
        run = AgentRunRecord("run")
        checkpoint = _checkpoint(1)
        engine = archive.state_store.storage_group.context.engine.sync_engine
        statements = []

        def count(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement.split()[0].upper())

        event.listen(engine, "before_cursor_execute", count)
        try:
            bundle = PreparedArchive()
            observation = await archive.transcript_repository.prepare_observation(
                "run", checkpoint.messages, first_message_index=0,
                pending=None, pending_keys=(), archive=bundle,
            )
            expected = await archive.prepare_checkpoints(
                run, (checkpoint,), observed_message_count=observation.target_message_count, archive=bundle,
            )
            head = await archive.execution_history_head_record("run")
            before = statements.count("SELECT")
            assert not set(statements) & {"INSERT", "UPDATE", "DELETE"}
            statements.clear()
            result = await archive.prepare_captured_projection(CapturedExecutionProjection(
                run, (), (checkpoint,), 0, 0, 0, 1,
                transcript=StagedTranscript(checkpoint.messages), target_message_index=1,
            ), baseline=None)
            assert result is not None
            after = statements.count("SELECT")
            assert not set(statements) & {"INSERT", "UPDATE", "DELETE"}
        finally:
            event.remove(engine, "before_cursor_execute", count)
        assert (before, after) == (3, 1)
        prepared, observed, _, bundle, captured_head = result
        assert captured_head == head
        assert observed is not None and observed.target_message_count == observation.target_message_count
        prepared = replace(prepared, checkpoints=tuple(
            replace(value, archive=bundle) for value in prepared.checkpoints
        ))
        await _assert_prepared_values_equal(archive, expected, prepared)
    finally:
        await state.close()


async def test_independent_registration_keeps_post_observation_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence._transcript_staging import StagedTranscript

    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="projection-order", tenant_id="tenant")
    try:
        archive = await _trace_archive(state)
        owner = archive._run_registrations
        assert owner is not None
        # The independent-owner path must never be enclosed by the domain read scope.
        monkeypatch.setattr(owner, "joins", lambda store: False)
        observed = []
        prepare_observation = archive.transcript_repository.prepare_observation
        prepare_registration = archive.prepare_run_registration
        prepare_checkpoints = archive.prepare_checkpoints

        async def observation(*args, **kwargs):
            observed.append("observation")
            return await prepare_observation(*args, **kwargs)

        async def registration(*args, **kwargs):
            assert active_state_scope() is None
            observed.append("registration")
            return await prepare_registration(*args, **kwargs)

        async def checkpoints(*args, **kwargs):
            observed.append("checkpoints")
            return await prepare_checkpoints(*args, **kwargs)

        monkeypatch.setattr(archive.transcript_repository, "prepare_observation", observation)
        monkeypatch.setattr(archive, "prepare_run_registration", registration)
        monkeypatch.setattr(archive, "prepare_checkpoints", checkpoints)
        await archive.prepare_captured_projection(CapturedExecutionProjection(
            AgentRunRecord("run"), (), (), 0, 0, 0, 0, transcript=StagedTranscript(()),
        ), baseline=None)
        assert observed == ["observation", "registration", "checkpoints"]
    finally:
        await state.close()


async def test_projection_preparation_honors_custom_single_run_head_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="projection-custom", tenant_id="tenant")
    try:
        archive = await _trace_archive(state)
        original = archive.execution_history_head_record
        read = []

        async def custom_head(self, agent_run_id):
            read.append(agent_run_id)
            return await original(agent_run_id)

        monkeypatch.setattr(type(archive), "execution_history_head_record", custom_head)
        await archive.prepare_captured_projection(CapturedExecutionProjection(
            AgentRunRecord("run"), (), (), 0, 0, 0, 0,
        ), baseline=None)
        assert read == ["run"]
    finally:
        await state.close()
