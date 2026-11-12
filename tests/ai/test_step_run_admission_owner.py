#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step admissions, not Runtime records, own local run publication."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai_harness.step_persistence import RunRecord
from sqlalchemy import text

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._official_step_reader import OfficialStepReader
from linktools.ai.persistence._run_registration import RunRegistrationReference
from linktools.ai.persistence.state import StateTransaction, StoredRecord

from .test_step_projection import _execution


@pytest.mark.asyncio
@pytest.mark.parametrize("sql", (False, True))
async def test_step_admission_is_local_publication_not_canonical_registration(tmp_path: Path, sql: bool) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "admissions.db") if sql else PersistenceStorage.in_memory()
    async with Persistence.open("admissions", tenant_id="tenant", storage=storage) as persistence:
        owner = persistence.run_store.run_registrations
        conversation = persistence.run_store.read_store(PersistenceDomain.CONVERSATION)
        trace = persistence.run_store.read_store(PersistenceDomain.TRACE)
        source = RunRecord(run_id="one", registration_id="registration", metadata={"canonical": "only"})
        await owner.publish(source)
        # Preparing a remote or local canonical registration is not publication
        # into either consuming domain, including after a process restart.
        assert await conversation.get_agent_run(agent_run_id="one") is None
        assert await trace.list_agent_runs() == []
        first = OfficialStepReader.project_run(source)
        second = replace(first, agent_conversation_id="override", parent_agent_run_id="parent",
                         metadata={"local": "projection"})
        await conversation.register_agent_run(first)
        await persistence.execution.executions.create_execution(_execution())
        await trace.register_agent_run(second, execution_id="execution")
        assert await conversation.get_agent_run(agent_run_id="one") == first
        assert await trace.list_agent_runs(agent_conversation_id="override") == [second]
        assert await trace.list_agent_runs(parent_agent_run_id="parent") == [second]
        assert await owner.get("one") == source
        if sql:
            async with conversation.state_store.context.engine.connect() as connection:
                assert await connection.scalar(text("SELECT count(*) FROM ai_state_records WHERE kind='agent_run'")) == 0
                assert await connection.scalar(text("SELECT count(*) FROM ai_step_run_admissions")) == 2
                assert await connection.scalar(text("SELECT count(*) FROM ai_step_runs")) == 1
        await conversation.release_agent_run("one")
        assert await conversation.get_agent_run(agent_run_id="one") is None
        assert await trace.get_agent_run(agent_run_id="one") == second
        assert await owner.get("one") == source
        await trace.release_agent_run("one", execution_id="execution")
        assert await trace.list_agent_runs() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("sql", (False, True))
async def test_failed_local_admission_transaction_never_publishes_run(tmp_path: Path, sql: bool) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "rollback.db") if sql else PersistenceStorage.in_memory()
    async with Persistence.open("admissions", tenant_id="tenant", storage=storage) as persistence:
        archive = persistence.run_store.read_store(PersistenceDomain.CONVERSATION)
        source = RunRecord(run_id="prepared", registration_id="registration")
        await persistence.run_store.run_registrations.publish(source)
        projection = OfficialStepReader.project_run(source)
        record = StoredRecord(archive._agent_run_key(source.run_id), None, None, "agent_run",
                              b"prepared", None, 0, None, 0, None,
                              RunRegistrationReference.capture(source, projection).to_data())

        async def abort(transaction: StateTransaction) -> None:
            await transaction.insert_record(record)
            raise AIError(ErrorCode.INTERNAL_ERROR)

        with pytest.raises(AIError) as caught:
            await archive.state_store.mutate(abort)
        assert caught.value.code is ErrorCode.INTERNAL_ERROR
        assert await archive.get_agent_run(agent_run_id="prepared") is None
        assert await archive.list_agent_runs() == []
        assert await persistence.run_store.run_registrations.get("prepared") == source
        if sql:
            async with archive.state_store.context.engine.connect() as connection:
                assert await connection.scalar(text("SELECT count(*) FROM ai_step_run_admissions")) == 0
                assert await connection.scalar(text("SELECT count(*) FROM ai_state_records WHERE kind='agent_run'")) == 0


@pytest.mark.asyncio
async def test_readonly_trace_snapshot_restores_only_step_admission_owner(tmp_path: Path) -> None:
    from linktools.ai.persistence._snapshot import SnapshotLimits
    from linktools.ai.persistence.snapshots import RuntimeSnapshotOperations
    from linktools.ai.storage import InMemoryObjectStore
    from .test_native_run_references import _source, _flush

    objects = InMemoryObjectStore()
    limits = SnapshotLimits(10000, 20_000_000)
    async with _source(tmp_path) as (state, _native, source, projection):
        await _flush(state)
        resources = state.snapshot_resources()
        resources = replace(resources, read_only=True, durable_domains=frozenset({PersistenceDomain.TRACE}),
                            stores={PersistenceDomain.TRACE: resources.stores[PersistenceDomain.TRACE]})
        snapshot = await RuntimeSnapshotOperations(resources, require_ready=lambda: None).export_snapshot(
            object_store=objects, limits=limits)
    destination = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=destination, limits=limits)
    async with Persistence.open("native-run-facts", tenant_id="tenant",
                                storage=PersistenceStorage.from_root(destination)) as restored:
        archive = restored.run_store.read_store(PersistenceDomain.TRACE)
        assert await archive.get_agent_run(agent_run_id="run") == projection
        assert await restored.run_store.run_registrations.get("run") == source
        async with archive.state_store.context.engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_state_records WHERE kind='agent_run'")) == 0
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_run_admissions")) == 1


@pytest.mark.asyncio
async def test_split_sql_admission_keeps_one_remote_canonical_and_fails_closed_on_cold_read(tmp_path: Path) -> None:
    from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive
    from linktools.ai.storage import InMemoryObjectStore

    async with Persistence.open("split", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "source.db")) as source:
        async with Persistence.open("split", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "target.db")) as target:
            owner = source.run_store.run_registrations
            store = target.run_store.read_store(PersistenceDomain.CONVERSATION).state_store

            def open_archive() -> ModelInteractionStateStepArchive:
                return ModelInteractionStateStepArchive(
                    store, object_store=InMemoryObjectStore(), namespace="split", tenant_id="tenant",
                    runtime_domain=PersistenceDomain.CONVERSATION, run_registrations=owner)

            archive = open_archive()
            await archive.initialize()
            run = RunRecord(run_id="split-run", registration_id="registration")
            await owner.publish(run)
            projection = OfficialStepReader.project_run(run)
            await archive.register_agent_run(projection)
            assert await archive.get_agent_run(agent_run_id="split-run") == projection
            async with store.context.engine.connect() as connection:
                assert await connection.scalar(text("SELECT count(*) FROM ai_state_records WHERE kind='agent_run'")) == 0
                assert await connection.scalar(text("SELECT count(*) FROM ai_step_run_admissions")) == 1
                assert await connection.scalar(text("SELECT count(*) FROM ai_step_runs")) == 0
            await archive.close()
            async with source.execution.executions.state_store.context.engine.begin() as connection:
                await connection.execute(text("DELETE FROM ai_step_runs"))
            cold = open_archive()
            await cold.initialize()
            try:
                with pytest.raises(AIError) as caught:
                    await cold.get_agent_run(agent_run_id="split-run")
                assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            finally:
                await cold.close()
