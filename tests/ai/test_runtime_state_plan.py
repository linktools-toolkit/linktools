#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime state routing defaults."""

import pytest

from linktools.ai.runtime import RuntimeDomain, RuntimeState, RuntimeStatePlan, RuntimeStateRoute


def test_runtime_state_sqlite_route_normalizes_paths(tmp_path) -> None:
    route = RuntimeStateRoute.sqlite(tmp_path / "runtime.db")

    assert route.path == (tmp_path / "runtime.db").resolve()
    assert RuntimeState.sqlite(tmp_path / "runtime.db").plan.durable_domains


def test_runtime_state_sqlite_uses_builtin_object_store_by_default(tmp_path) -> None:
    state = RuntimeState.sqlite(tmp_path / "runtime.db")

    assert state.plan.durable_domains


def test_runtime_state_plan_allows_sqlite_without_an_explicit_object_store(
    tmp_path,
) -> None:
    route = RuntimeStateRoute.sqlite(tmp_path / "runtime.db")
    plan = RuntimeStatePlan(
        **{
            domain.value: route
            for domain in RuntimeDomain
            if domain is not RuntimeDomain.RECOVERY
        }
    )

    assert RuntimeState.from_plan(plan).plan == plan


def test_default_state_plan_uses_memory_for_all_domains() -> None:
    plan = RuntimeStatePlan()

    assert all(plan.route(domain).kind == "memory" for domain in RuntimeDomain)


@pytest.mark.asyncio
async def test_filesystem_plan_rejects_effective_member_path_overlap(tmp_path) -> None:
    execution_root = tmp_path / "state"
    plan = RuntimeStatePlan(
        execution=RuntimeStateRoute.filesystem(execution_root),
        memory=RuntimeStateRoute.filesystem(execution_root / "execution"),
    )
    state = RuntimeState.from_plan(plan)

    with pytest.raises(ValueError, match="member paths overlap"):
        await state.initialize(namespace="runtime", tenant_id="tenant")

    assert not execution_root.exists()


@pytest.mark.asyncio
async def test_filesystem_execution_and_recovery_share_object_store(tmp_path) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="runtime", tenant_id="tenant")
    try:
        assert (
            state.object_store(RuntimeDomain.EXECUTION)
            is state.object_store(RuntimeDomain.RECOVERY)
        )
    finally:
        await state.close()


def test_durable_evaluation_requires_durable_execution(tmp_path) -> None:
    plan = RuntimeStatePlan(
        evaluation=RuntimeStateRoute.filesystem(tmp_path / "evaluation"),
    )

    with pytest.raises(
        ValueError,
        match="durable evaluation requires durable execution",
    ):
        RuntimeState.from_plan(plan)
