#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current model binding and RuntimeState contract checks."""

import inspect

from linktools.ai.model import ModelRegistry
from linktools.ai.runtime.state import RuntimeDomain, RuntimeState, RuntimeStatePlan, RuntimeStateRoute


def test_model_binding_fingerprint_excludes_secret_material() -> None:
    registry = ModelRegistry.openai(model="gpt-test", api_key="secret")
    binding = registry.snapshot().resolve("default")
    assert "secret" not in repr(binding)
    assert "secret" not in binding.fingerprint
    assert binding.model_identity == "openai:gpt-test"


def test_runtime_state_sqlite_route_normalizes_paths(tmp_path) -> None:
    route = RuntimeStateRoute.sqlite(tmp_path / "runtime.db")
    assert route.path == (tmp_path / "runtime.db").resolve()
    assert RuntimeState.sqlite(tmp_path / "runtime.db").plan.durable_domains


def test_public_opener_does_not_expose_removed_storage_argument() -> None:
    from linktools.ai.workspace import open_workspace_runtime

    parameters = inspect.signature(open_workspace_runtime).parameters
    assert "storage_root" not in parameters
    assert "runtime_storage" not in parameters


def test_default_state_plan_has_all_domains() -> None:
    plan = RuntimeStatePlan()
    assert all(plan.route(domain).kind == "memory" for domain in RuntimeDomain)
