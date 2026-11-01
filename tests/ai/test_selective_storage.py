#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PersistenceStorage route and persistence checks."""


from linktools.ai.persistence import PersistenceDomain, PersistencePlan
from linktools.ai.storage import StorageRoute


def test_runtime_storage_plan_routes_each_domain_explicitly(tmp_path) -> None:
    transaction_root = tmp_path / "state"
    conversation_root = transaction_root / "conversation.db"
    plan = PersistencePlan(
        conversation=StorageRoute.sqlite(conversation_root),
        execution=StorageRoute.sqlite(transaction_root / "execution.db"),
    )
    assert plan.route(PersistenceDomain.CONVERSATION).path == conversation_root.resolve()
    assert plan.route(PersistenceDomain.EXECUTION).retention.value == "durable"
    assert plan.route(PersistenceDomain.MEMORY).kind == "memory"
