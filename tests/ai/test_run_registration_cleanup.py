#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared run metadata is reclaimed only with complete offline source authority."""

from contextlib import asynccontextmanager

import pytest
from pydantic_ai_harness.step_persistence import RunRecord

from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive
from linktools.ai.persistence._official_step_reader import OfficialStepReader
from linktools.ai.storage import InMemoryObjectStore


class _ExclusiveSources:
    def __init__(self, expected):
        self.expected = set(expected)
        self.entered = False
        self.calls = 0

    @asynccontextmanager
    async def __call__(self, groups):
        # These tests have no concurrent source users. A real caller must drain
        # and exclude readers, snapshot exporters, and writers across complete
        # prepare-through-publication operations, including external borrowers,
        # before yielding and until exit. This assertion cannot enforce that
        # application-owned exclusion; it checks only the requested group set.
        assert set(groups) == self.expected
        assert not self.entered
        self.entered = True
        self.calls += 1
        try:
            yield
        finally:
            self.entered = False


@pytest.mark.asyncio
async def test_public_offline_cleanup_preserves_external_archive_then_reclaims_released_run() -> None:
    async with Persistence.open("cleanup", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        resources = state.snapshot_resources()
        owner = state.run_store.run_registrations
        archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        external_state = InMemoryStateStore()
        await external_state.initialize()
        external = ModelInteractionStateStepArchive(
            external_state, object_store=InMemoryObjectStore(), namespace="cleanup", tenant_id="tenant",
            runtime_domain=PersistenceDomain.CONVERSATION, run_registrations=owner,
        )
        await external.initialize()
        source = RunRecord(run_id="run", registration_id="registration", metadata={"owner": "shared"})
        await owner.publish(source)
        projected = OfficialStepReader.project_run(source)
        await archive.register_agent_run(projected)
        await external.register_agent_run(projected)
        await archive.release_agent_run("run")
        assert await owner.get("run") == source
        inventory = (*resources.stores.values(), external_state)
        guard = _ExclusiveSources(store.storage_group for store in inventory)
        operations = state.snapshot_operations()
        try:
            # Neither the ordinary release nor an omitted completeness claim
            # authorizes deletion of metadata another archive still references.
            assert (await operations.purge_unreferenced_run_registrations(
                ("run",), exclusive=guard,
            )).blocked == ("run",)
            assert guard.calls == 0
            assert (await operations.purge_unreferenced_run_registrations(
                ("run",), exclusive=guard, complete_reference_stores=inventory,
            )).retained == ("run",)
            assert await external.get_agent_run(agent_run_id="run") == projected
            await external.release_agent_run("run")
            result = await operations.purge_unreferenced_run_registrations(
                ("run", "missing", "run"), exclusive=guard, complete_reference_stores=inventory,
            )
            assert result.deleted == ("run",) and result.missing == ("missing",)
            assert await owner.get("run") is None
            assert guard.calls == 2 and not guard.entered
        finally:
            await external.close()
            await external_state.close()


@pytest.mark.asyncio
async def test_offline_cleanup_blocks_incomplete_known_inventory_and_old_guard() -> None:
    async with Persistence.open("cleanup", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        owner = state.run_store.run_registrations
        await owner.publish(RunRecord(run_id="run"))
        operations = state.snapshot_operations()
        guard = _ExclusiveSources(())
        for inventory in (None, ()):
            assert (await operations.purge_unreferenced_run_registrations(
                ("run",), exclusive=guard, complete_reference_stores=inventory,
            )).blocked == ("run",)
        inventory = tuple(state.snapshot_resources().stores.values())
        assert (await operations.purge_unreferenced_run_registrations(
            ("run",), exclusive=object(), complete_reference_stores=inventory,
        )).blocked == ("run",)
        assert await owner.get("run") is not None and guard.calls == 0


@pytest.mark.asyncio
async def test_offline_cleanup_never_deletes_sql_registration_only_headers(tmp_path) -> None:
    async with Persistence.open("cleanup", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "runs.db")) as state:
        owner = state.run_store.run_registrations
        source = RunRecord(run_id="run")
        await owner.publish(source)
        inventory = tuple(state.snapshot_resources().stores.values())
        guard = _ExclusiveSources(store.storage_group for store in inventory)
        result = await state.snapshot_operations().purge_unreferenced_run_registrations(
            ("run",), exclusive=guard, complete_reference_stores=inventory,
        )
        assert result.blocked == ("run",) and guard.calls == 0
        assert await owner.get("run") == source


@pytest.mark.asyncio
async def test_offline_cleanup_rejects_unquiesced_groups_and_bounds_each_batch() -> None:
    async with Persistence.open("cleanup", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        owner = state.run_store.run_registrations
        archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        identities = ("first", "second", "third")
        for identity in identities:
            run = OfficialStepReader.project_run(RunRecord(run_id=identity))
            await archive.register_agent_run(run)
            await archive.release_agent_run(identity)
        inventory = tuple(state.snapshot_resources().stores.values())
        calls = []

        @asynccontextmanager
        async def cannot_exclude(groups):
            calls.append(groups)
            raise RuntimeError("external source still has active users")
            yield

        with pytest.raises(RuntimeError, match="active users"):
            await state.snapshot_operations().purge_unreferenced_run_registrations(
                identities, exclusive=cannot_exclude, complete_reference_stores=inventory,
            )
        assert len(calls) == 1 and set(await owner.get_many(identities)) == set(identities)
        guard = _ExclusiveSources(store.storage_group for store in inventory)
        result = await state.snapshot_operations().purge_unreferenced_run_registrations(
            identities, exclusive=guard, complete_reference_stores=inventory, limit=2,
        )
        assert result.deleted == identities[:2] and result.blocked == identities[2:]
        assert set(await owner.get_many(identities)) == {"third"}
