#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime state routing defaults."""

from linktools.ai.runtime import RuntimeDomain, RuntimeState, RuntimeStatePlan, RuntimeStateRoute


def test_runtime_state_sqlite_route_normalizes_paths(tmp_path) -> None:
    route = RuntimeStateRoute.sqlite(tmp_path / "runtime.db")

    assert route.path == (tmp_path / "runtime.db").resolve()
    assert RuntimeState.sqlite(tmp_path / "runtime.db").plan.durable_domains


def test_default_state_plan_uses_memory_for_all_domains() -> None:
    plan = RuntimeStatePlan()

    assert all(plan.route(domain).kind == "memory" for domain in RuntimeDomain)
