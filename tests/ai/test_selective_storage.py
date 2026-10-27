#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PersistenceStorage route and persistence checks."""

import json

import pytest
from linktools.ai.persistence import PersistenceDomain, PersistenceStorage, PersistencePlan
from linktools.ai.persistence import Persistence
from linktools.ai.storage import StorageRoute


def test_runtime_storage_plan_routes_each_domain_explicitly(tmp_path) -> None:
    transaction_root = tmp_path / "state"
    conversation_root = transaction_root / "conversation"
    plan = PersistencePlan(
        conversation=StorageRoute.filesystem(
            conversation_root,
            transaction_root=transaction_root,
        ),
        execution=StorageRoute.filesystem(
            transaction_root,
            transaction_root=transaction_root,
        ),
    )
    assert plan.route(PersistenceDomain.CONVERSATION).path == conversation_root.resolve()
    assert plan.route(PersistenceDomain.EXECUTION).retention.value == "durable"
    assert plan.route(PersistenceDomain.MEMORY).kind == "memory"


@pytest.mark.asyncio
async def test_filesystem_state_writes_domain_manifest(tmp_path) -> None:
    root = tmp_path / "runtime"
    state = Persistence(PersistenceStorage.from_plan(
        PersistencePlan(
            conversation=StorageRoute.filesystem(
                root / "conversation",
                transaction_root=root,
            ),
            execution=StorageRoute.filesystem(
                root,
                transaction_root=root,
            ),
        )
    ))
    await state.initialize(namespace="selective", tenant_id="tenant")
    try:
        manifests = list((root / "conversation").rglob("manifest.json"))
        assert len(manifests) == 1
        assert json.loads(manifests[0].read_text(encoding="utf-8"))["format"] == "linktools-ai-state"
    finally:
        await state.close()
