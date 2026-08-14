#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeState resource ownership and restart checks."""

import asyncio

import pytest

from linktools.ai.runtime.state import RuntimeDomain, RuntimeState, RuntimeStatePlan, RuntimeStateRoute


@pytest.mark.asyncio
async def test_sqlite_runtime_state_owns_and_reopens_database(tmp_path) -> None:
    path = tmp_path / "runtime.db"
    state = RuntimeState.sqlite(path)
    await state.initialize(namespace="runtime", tenant_id="tenant")
    assert state.lifecycle == "ready"
    await state.close()

    reopened = RuntimeState.sqlite(path)
    await reopened.initialize(namespace="runtime", tenant_id="tenant")
    assert reopened.lifecycle == "ready"
    await reopened.close()


def test_mixed_runtime_plan_has_explicit_routes(tmp_path) -> None:
    plan = RuntimeStatePlan(
        conversation=RuntimeStateRoute.filesystem(tmp_path / "conversation"),
        execution=RuntimeStateRoute.transient(),
        memory=RuntimeStateRoute.memory(),
    )
    assert plan.route(RuntimeDomain.CONVERSATION).retention.value == "durable"
    assert plan.route(RuntimeDomain.EXECUTION).retention.value == "transient"
    assert plan.durable_domains == frozenset({RuntimeDomain.CONVERSATION})
