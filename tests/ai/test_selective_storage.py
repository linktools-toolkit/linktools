#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeStorage route and persistence checks."""

import json

import pytest
from linktools.ai.runtime.state import (
    RuntimeDomain,
    RuntimeStorage,
    RuntimeStoragePlan,
    RuntimeStorageRoute,
)


def test_runtime_storage_plan_routes_each_domain_explicitly(tmp_path) -> None:
    transaction_root = tmp_path / "state"
    conversation_root = transaction_root / "conversation"
    plan = RuntimeStoragePlan(
        conversation=RuntimeStorageRoute.filesystem(
            conversation_root,
            transaction_root=transaction_root,
        ),
        execution=RuntimeStorageRoute.filesystem(
            transaction_root,
            transaction_root=transaction_root,
        ),
    )
    assert plan.route(RuntimeDomain.CONVERSATION).path == conversation_root.resolve()
    assert plan.route(RuntimeDomain.EXECUTION).retention.value == "durable"
    assert plan.route(RuntimeDomain.MEMORY).kind == "memory"


@pytest.mark.asyncio
async def test_filesystem_state_writes_domain_manifest(tmp_path) -> None:
    root = tmp_path / "runtime"
    state = RuntimeStorage.from_plan(
        RuntimeStoragePlan(
            conversation=RuntimeStorageRoute.filesystem(
                root / "conversation",
                transaction_root=root,
            ),
            execution=RuntimeStorageRoute.filesystem(
                root,
                transaction_root=root,
            ),
        )
    )
    await state.initialize(namespace="selective", tenant_id="tenant")
    try:
        manifests = list((root / "conversation").rglob("manifest.json"))
        assert len(manifests) == 1
        assert json.loads(manifests[0].read_text(encoding="utf-8"))["format"] == "linktools-ai-state"
    finally:
        await state.close()
