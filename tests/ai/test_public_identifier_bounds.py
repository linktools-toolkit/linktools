#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atomic identity byte budgets do not restrict compound or logical identities."""

from collections.abc import Callable
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import mysql, postgresql, sqlite

from linktools.ai.core import (
    HmacCursorSigner, Principal, TenantAuthorizationPolicy,
    validate_agent_id, validate_capability_id, validate_capability_write_key,
    validate_idempotency_key, validate_lease_owner, validate_logical_id,
    validate_memory_scope, validate_persistence_namespace, validate_principal_id,
    validate_resource_id, validate_tenant_id,
)
from linktools.ai.errors import AIError
from linktools.ai.observe import Metrics
from linktools.ai.observe._sql import build_metrics_sql_metadata
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.session import CloseSessionRequest, SessionManager

from .test_session_timeline import _session


@pytest.mark.parametrize("validate", (
    validate_persistence_namespace, validate_memory_scope, validate_principal_id,
    validate_capability_id, validate_capability_write_key, validate_idempotency_key,
    validate_resource_id, validate_tenant_id,
))
@pytest.mark.parametrize("value", ("x" * 128, "界" * 42 + "ab"))
def test_atomic_identifiers_preserve_128_bytes_and_reject_129(validate: Callable[[str], str], value: str) -> None:
    assert validate(value) == value
    with pytest.raises(AIError):
        validate(value + "x")


@pytest.mark.parametrize("validate", (validate_logical_id, validate_agent_id))
def test_logical_identity_limit_remains_64_characters(validate: Callable[[str], str]) -> None:
    value = "界" * 64
    assert validate(value) == value
    with pytest.raises((AIError, ValueError)):
        validate(value + "界")


def test_lease_owner_retains_compound_identity_budget() -> None:
    owner = "task-engine:" + "界" * 42 + "ab:" + "f" * 32
    assert len(owner.encode("utf-8")) == 173
    assert validate_lease_owner(owner) == owner
    assert validate_lease_owner("x" * 256) == "x" * 256
    with pytest.raises(AIError):
        validate_lease_owner("x" * 257)


def test_metrics_namespace_matches_atomic_input_and_all_sql_dialects() -> None:
    namespace = "界" * 42 + "ab"
    assert Metrics.in_memory(namespace=namespace).namespace == namespace
    with pytest.raises(AIError):
        Metrics.in_memory(namespace=namespace + "x")
    metadata = build_metrics_sql_metadata()
    for name in ("ai_metric_definitions", "ai_metric_observations"):
        column = metadata.tables[name].c.namespace
        for dialect in (mysql.dialect(), postgresql.dialect(), sqlite.dialect()):
            assert column.type.dialect_impl(dialect).length == 128


@pytest.mark.asyncio
async def test_force_close_child_keys_preserve_parent_and_target_identity() -> None:
    principal = Principal("owner", "tenant")
    keys = []
    target = "e" * 128

    class CancelledAtBoundary(Exception):
        pass

    class Execution:
        async def cancel(self, execution_id: str, *, principal: Principal, idempotency_key: str) -> None:
            validate_idempotency_key(idempotency_key)
            keys.append(idempotency_key)
            raise CancelledAtBoundary

    async def pending(session_id: str, *, tenant_id: str) -> SimpleNamespace:
        return SimpleNamespace(execution_id=target)

    async with Persistence.open("session-bounds", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        await state.conversation.sessions.create(_session())
        manager = SessionManager(
            state.conversation, state.execution.executions,
            TenantAuthorizationPolicy(), Execution(),
            HmacCursorSigner("session-bounds", b"session-bounds-key"),
            pending_session_execution=pending,
        )
        parent = "界" * 42 + "ab"
        for key in (parent, parent, "界" * 42 + "ac"):
            with pytest.raises(CancelledAtBoundary):
                await manager.close("session", CloseSessionRequest(principal, key, force=True))
        target = "f" * 128
        with pytest.raises(CancelledAtBoundary):
            await manager.close("session", CloseSessionRequest(principal, parent, force=True))
    assert all(len(key) == 36 for key in keys)
    assert keys[0] == keys[1]
    assert len({keys[0], keys[2], keys[3]}) == 3
