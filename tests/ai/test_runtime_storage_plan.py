#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime storage routing defaults."""

import pytest

from linktools.ai.persistence import PersistenceDomain, PersistenceStorage, PersistencePlan
from linktools.ai.persistence import Persistence
from linktools.ai.storage import StorageRoute


def test_runtime_storage_sqlite_route_normalizes_paths(tmp_path) -> None:
    route = StorageRoute.sqlite(tmp_path / "runtime.db")

    assert route.path == (tmp_path / "runtime.db").resolve()
    assert PersistenceStorage.sqlite(tmp_path / "runtime.db").plan.durable_domains


def test_runtime_storage_plan_allows_sqlite_without_an_explicit_object_store(
    tmp_path,
) -> None:
    route = StorageRoute.sqlite(tmp_path / "runtime.db")
    plan = PersistencePlan(
        **{
            domain.value: route
            for domain in PersistenceDomain
            if domain not in {PersistenceDomain.TRACE, PersistenceDomain.CHECKPOINT}
        }
    )

    assert PersistenceStorage.from_plan(plan).plan == plan


def test_default_state_plan_uses_memory_for_all_domains() -> None:
    plan = PersistencePlan()

    assert all(plan.route(domain).kind == "memory" for domain in PersistenceDomain)




@pytest.mark.asyncio
async def test_local_sqlite_execution_and_recovery_share_object_store(tmp_path) -> None:
    state = Persistence(PersistenceStorage.from_root(tmp_path / "runtime", exclusive_writer=True))
    await state.initialize(namespace="runtime", tenant_id="tenant")
    try:
        assert (
            state.object_store(PersistenceDomain.EXECUTION)
            is state.object_store(PersistenceDomain.CHECKPOINT)
        )
    finally:
        await state.close()


def test_durable_evaluation_requires_durable_execution(tmp_path) -> None:
    plan = PersistencePlan(
        evaluation=StorageRoute.sqlite(tmp_path / "evaluation.db"),
    )

    with pytest.raises(
        ValueError,
        match="durable evaluation requires durable execution",
    ):
        PersistenceStorage.from_plan(plan)

