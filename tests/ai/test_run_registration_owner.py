#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run metadata has one owner independently of optional Step capture."""

import asyncio
from dataclasses import replace
from datetime import timedelta

import pytest
from pydantic_ai_harness.step_persistence import InMemoryStepStore, RunRecord

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._run_registration import RunRegistrationStore, RunRegistrationReference
from linktools.ai.persistence._official_step_reader import OfficialStepReader


@pytest.mark.asyncio
async def test_confirmed_external_registration_survives_external_store_lifetime() -> None:
    state = InMemoryStateStore()
    await state.initialize()
    external = InMemoryStepStore()
    source = RunRecord(run_id="run", registration_id="registration", metadata={"source": "native"})
    await external.register_run(source)
    confirmed = await external.get_run(run_id="run")
    assert confirmed is not None
    owner = RunRegistrationStore(state, namespace="owner", tenant_id="tenant")
    await owner.publish(confirmed)
    external = InMemoryStepStore()
    assert await external.get_run(run_id="run") is None
    reopened = RunRegistrationStore(state, namespace="owner", tenant_id="tenant")
    assert await reopened.get("run") == source


@pytest.mark.asyncio
async def test_domains_import_one_equal_registration_and_reject_conflicts() -> None:
    state = InMemoryStateStore()
    await state.initialize()
    owners = [RunRegistrationStore(state, namespace="owner", tenant_id="tenant") for _ in range(3)]
    source = RunRecord(run_id="run", registration_id="registration", metadata={"shared": "value"})
    await asyncio.gather(*(owner.publish(source) for owner in owners))
    for owner in owners:
        assert await owner.get("run") == source
    with pytest.raises(AIError) as caught:
        await owners[0].publish(replace(source, metadata={"shared": "conflict"}))
    assert caught.value.code is ErrorCode.STORAGE_CONFLICT
    assert await owners[0].get("run") == source


@pytest.mark.asyncio
async def test_run_registration_is_partitioned_by_namespace_and_tenant() -> None:
    state = InMemoryStateStore()
    await state.initialize()
    owner = RunRegistrationStore(state, namespace="owner", tenant_id="tenant")
    await owner.publish(RunRecord(run_id="run"))
    assert await RunRegistrationStore(state, namespace="other", tenant_id="tenant").get("run") is None
    assert await RunRegistrationStore(state, namespace="owner", tenant_id="other").get("run") is None


def test_domain_reference_stores_only_projection_differences() -> None:
    source = RunRecord(run_id="run", registration_id="registration", metadata={"same": "large" * 1000,
                        "different": "source", "removed": "value"})
    projected = replace(OfficialStepReader.project_run(source),
                        metadata={"same": source.metadata["same"], "different": "domain", "added": "value"},
                        started_at=source.started_at + timedelta(microseconds=1))
    reference = RunRegistrationReference.capture(source, projected)
    assert reference.metadata == {"different": "domain", "added": "value"}
    assert reference.removed_metadata == ("removed",)
    assert reference.project(source) == projected
    with pytest.raises(AIError) as caught:
        reference.project(replace(source, registration_id="another"))
    assert caught.value.code is ErrorCode.STORAGE_OWNER_MISMATCH


def test_equal_projection_has_no_duplicate_fields_or_metadata() -> None:
    source = RunRecord(run_id="run", metadata={"shared": "value"})
    reference = RunRegistrationReference.capture(source, OfficialStepReader.project_run(source))
    assert reference.fields == {} and reference.metadata == {} and reference.removed_metadata == ()
    assert reference.project(source) == OfficialStepReader.project_run(source)
    from linktools.ai.persistence.codec import decode_envelope

    assert "projection_digest" not in decode_envelope(reference.to_data()).value["payload"]
    restored = RunRegistrationReference.from_data(reference.to_data())
    accepted_source = replace(source, metadata={"shared": "canonical owner value"})
    assert restored.project(accepted_source) == OfficialStepReader.project_run(accepted_source)


@pytest.mark.asyncio
@pytest.mark.parametrize("same_group", (False, True))
async def test_cold_domain_reads_share_registration_without_nested_owner_scope(same_group: bool) -> None:
    from linktools.ai.core import PersistenceDomain
    from linktools.ai.persistence._memory import InMemoryStateStorageGroup
    from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive
    from linktools.ai.storage import InMemoryObjectStore

    group = InMemoryStateStorageGroup()
    source_state = InMemoryStateStore(group)
    domain_state = InMemoryStateStore(group if same_group else None)
    await source_state.initialize()
    await domain_state.initialize()
    source = RunRecord(run_id="shared", metadata={"source": "canonical"})
    owner = RunRegistrationStore(source_state, namespace="owner", tenant_id="tenant")
    await owner.publish(source)

    def open_archive() -> ModelInteractionStateStepArchive:
        return ModelInteractionStateStepArchive(
            domain_state, object_store=InMemoryObjectStore(), namespace="owner", tenant_id="tenant",
            runtime_domain=PersistenceDomain.CONVERSATION, run_registrations=owner,
        )

    archive = open_archive()
    await archive.initialize()
    projected = OfficialStepReader.project_run(source)
    await archive.register_agent_run(projected)
    await archive.close()
    reopened = open_archive()
    await reopened.initialize()
    try:
        assert await reopened.get_agent_run(agent_run_id="shared") == projected
        assert await reopened.list_agent_runs() == [projected]
        run, interactions = await reopened.read_model_interactions_with_run(agent_run_id="shared")
        assert run == projected and interactions == []
        assert await owner.get("shared") == source
    finally:
        await reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_start", (False, True))
async def test_registration_has_final_seed_before_domain_admission(shared_start: bool) -> None:
    from datetime import datetime, timezone
    from linktools.ai.capability.steps import StepRunRegistration

    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    async def seed():
        return "durable-registration", timestamp

    registration = StepRunRegistration(run_id="run", parent_run_id="parent", agent_name="agent",
                                       metadata={"value": "source"}, share_started=shared_start)
    registration.bind_registration(seed)
    admitted = await registration.prepare_registration(conversation_id="conversation")
    assert admitted.registration_id == "durable-registration" and admitted.started_at == timestamp
    if shared_start:
        native, event = await registration.prepare_started(conversation_id="conversation", step_index=0)
        assert event.timestamp == timestamp
    else:
        native = await registration.prepare_registration(conversation_id="conversation")
    assert native == admitted
    with pytest.raises(ValueError):
        registration.prepare(conversation_id="conversation", registration_id="different-registration")


@pytest.mark.asyncio
async def test_cold_sql_conversation_run_uses_one_connection_without_event_tables(tmp_path) -> None:
    from sqlalchemy import inspect
    from sqlalchemy.ext.asyncio import create_async_engine
    from linktools.ai.core import PersistenceDomain
    from linktools.ai.migrate import provision_runtime_database
    from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive
    from linktools.ai.persistence._sql import SqlStateStore
    from linktools.ai.persistence._step_store import SqlStepStore
    from linktools.ai.storage import InMemoryObjectStore

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'single-connection.db'}",
        pool_size=1, max_overflow=0, pool_timeout=0.1,
    )
    state = SqlStateStore(engine, runtime_domain=PersistenceDomain.CONVERSATION)
    archive = None
    try:
        await provision_runtime_database(engine, domains={PersistenceDomain.CONVERSATION})
        async with engine.connect() as connection:
            tables = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
        assert "ai_step_runs" in tables and "ai_step_events" not in tables
        await state.initialize()
        native = SqlStepStore(state.context, namespace="owner", tenant_id="tenant")
        owner = RunRegistrationStore(state, namespace="owner", tenant_id="tenant", native=native)

        def open_archive():
            return ModelInteractionStateStepArchive(
                state, object_store=InMemoryObjectStore(), namespace="owner", tenant_id="tenant",
                runtime_domain=PersistenceDomain.CONVERSATION, run_registrations=owner,
            )

        source = RunRecord(run_id="cold", metadata={"source": "canonical"})
        projected = OfficialStepReader.project_run(source)
        archive = open_archive()
        await archive.initialize()
        await archive.register_agent_run(projected)
        await archive.close()
        archive = open_archive()
        await archive.initialize()
        assert not archive.can_share_native_start(native)
        assert await archive.get_agent_run(agent_run_id="cold") == projected
        assert await archive.list_agent_runs() == [projected]
        run, interactions = await archive.read_model_interactions_with_run(agent_run_id="cold")
        assert run == projected and interactions == []
    finally:
        if archive is not None:
            await archive.close()
        await state.close()
        await engine.dispose()
