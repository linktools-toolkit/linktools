#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeState route and ownership checks."""

import inspect
import json

import pytest
from linktools.ai import Runtime
from linktools.ai.runtime.state import (
    RuntimeDomain,
    RuntimeState,
    RuntimeStatePlan,
    RuntimeStateRoute,
)


def test_runtime_state_plan_routes_each_domain_explicitly(tmp_path) -> None:
    conversation_root = tmp_path / "conversation"
    plan = RuntimeStatePlan(
        conversation=RuntimeStateRoute.filesystem(conversation_root),
        execution=RuntimeStateRoute.transient(),
    )
    assert plan.route(RuntimeDomain.CONVERSATION).path == conversation_root.resolve()
    assert plan.route(RuntimeDomain.EXECUTION).retention.value == "transient"
    assert plan.route(RuntimeDomain.MEMORY).kind == "memory"


@pytest.mark.asyncio
async def test_filesystem_state_writes_domain_manifest(tmp_path) -> None:
    root = tmp_path / "runtime"
    state = RuntimeState.from_plan(
        RuntimeStatePlan(
            conversation=RuntimeStateRoute.filesystem(root / "conversation"),
            execution=RuntimeStateRoute.filesystem(root / "execution"),
            recovery=RuntimeStateRoute.filesystem(root / "recovery"),
        )
    )
    await state.initialize(namespace="selective", tenant_id="tenant")
    try:
        manifests = list((root / "conversation").rglob("manifest.json"))
        assert len(manifests) == 1
        assert json.loads(manifests[0].read_text(encoding="utf-8"))["format"] == "linktools-ai-state"
    finally:
        await state.close()


def test_public_runtime_surface_is_not_storage_composition() -> None:
    parameters = inspect.signature(Runtime.open).parameters
    assert "state" in parameters
    assert "runtime_storage" not in parameters
    assert "storage_root" not in parameters
