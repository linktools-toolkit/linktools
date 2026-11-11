#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session metadata uses one JSON-object projection across validation and receipts."""

import asyncio
from collections import UserDict
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from types import MappingProxyType

import pytest

from linktools.ai.core import (
    AuthorizationAction,
    HmacCursorSigner,
    JsonValue,
    Principal,
    ResourceRef,
    TenantAuthorizationPolicy,
    canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.session import CreateSessionRequest, SessionManager, UpdateSessionRequest

pytestmark = pytest.mark.asyncio
_PRINCIPAL = Principal("owner", "tenant")


@pytest.fixture(params=("memory", "sqlite"))
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> PersistenceStorage:
    return (PersistenceStorage.in_memory() if request.param == "memory"
            else PersistenceStorage.sqlite(tmp_path / "state.db"))


def _metadata(
    kind: str, values: dict[str, JsonValue],
) -> tuple[Mapping[str, JsonValue], MutableMapping[str, JsonValue]]:
    if kind == "mappingproxy":
        return MappingProxyType(values), values
    if kind == "userdict":
        value = UserDict(values)
        return value, value
    return values, values


def _manager(state: Persistence) -> SessionManager[object]:
    return SessionManager(
        state.conversation, state.execution.executions,
        TenantAuthorizationPolicy(), object(),
        HmacCursorSigner("session-metadata", b"session-metadata-test-key"),
    )


@pytest.mark.parametrize("kind", ("dict", "mappingproxy", "userdict"))
async def test_session_metadata_mappings_share_request_identity(
    storage: PersistenceStorage, kind: str,
) -> None:
    async with Persistence.open("session-metadata", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        initial, _ = _metadata(kind, {"label": "initial"})
        await manager.create("agent", CreateSessionRequest(
            _PRINCIPAL, "session", "create", metadata=initial))
        values: dict[str, JsonValue] = {"label": "updated", "nested": {"items": [1, True, None]}}
        metadata, _ = _metadata(kind, values)
        updated = await manager.update("agent", "session", UpdateSessionRequest(
            _PRINCIPAL, 0, "update", metadata))
        assert updated.metadata == values
        assert updated.revision == 1
        receipt = await state.conversation.operations.get(
            "update", tenant_id="tenant")
        assert receipt is not None
        assert receipt.request_digest == canonical_sha256({
            "action": "session.update", "tenant_id": "tenant", "principal_id": "owner",
            "session_id": "session", "expected_revision": 0, "metadata": values, "cwd": None,
        })
        for replay_kind in ("dict", "mappingproxy", "userdict"):
            replay, _ = _metadata(replay_kind, dict(values))
            assert await manager.update("agent", "session", UpdateSessionRequest(
                _PRINCIPAL, 0, "update", replay)) == updated
            assert await state.conversation.operations.get(
                receipt.operation_id, tenant_id="tenant") == receipt
        persisted = await state.conversation.sessions.get("session", tenant_id="tenant")
        assert persisted is not None
        assert persisted.metadata == values and persisted.revision == 1


@pytest.mark.parametrize("kind", ("dict", "mappingproxy", "userdict"))
async def test_session_metadata_mapping_keeps_request_rejections(
    storage: PersistenceStorage, kind: str,
) -> None:
    async with Persistence.open("session-metadata", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        await manager.create("agent", CreateSessionRequest(_PRINCIPAL, "session", "create"))
        values, _ = _metadata(kind, {"label": "updated"})
        updated = await manager.update("agent", "session", UpdateSessionRequest(
            _PRINCIPAL, 0, "update", values))
        receipt = await state.conversation.operations.get(
            "update", tenant_id="tenant")

        changed, _ = _metadata(kind, {"label": "different"})
        with pytest.raises(AIError) as conflict:
            await manager.update("agent", "session", UpdateSessionRequest(
                _PRINCIPAL, 0, "update", changed))
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT

        with pytest.raises(AIError) as stale:
            await manager.update("agent", "session", UpdateSessionRequest(
                _PRINCIPAL, 0, "stale", values))
        assert stale.value.code is ErrorCode.SESSION_REVISION_CONFLICT

        reserved, _ = _metadata(kind, {"linktools.ai.reserved": True})
        with pytest.raises(AIError) as invalid:
            await manager.update("agent", "session", UpdateSessionRequest(
                _PRINCIPAL, updated.revision, "reserved", reserved))
        assert invalid.value.code is ErrorCode.REQUEST_FIELD_INVALID

        assert await manager.get("session", principal=_PRINCIPAL) == updated
        assert await state.conversation.operations.get(
            "update", tenant_id="tenant") == receipt
        for key in ("stale", "reserved"):
            assert await state.conversation.operations.get(
                key, tenant_id="tenant") is None


@pytest.mark.parametrize("kind", ("dict", "mappingproxy", "userdict"))
async def test_session_update_snapshots_outer_metadata_before_cwd_await(
    storage: PersistenceStorage, kind: str,
) -> None:
    entered, proceed = asyncio.Event(), asyncio.Event()

    async def canonicalize_path(value: str) -> str:
        entered.set()
        await proceed.wait()
        return value

    async with Persistence.open("session-metadata", storage=storage, tenant_id="tenant") as state:
        manager = SessionManager(
            state.conversation, state.execution.executions,
            TenantAuthorizationPolicy(), object(),
            HmacCursorSigner("session-metadata", b"session-metadata-test-key"),
            canonicalize_path=canonicalize_path,
        )
        await manager.create("agent", CreateSessionRequest(_PRINCIPAL, "session", "create"))
        metadata, mutable = _metadata(kind, {"label": "before"})
        task = asyncio.create_task(manager.update("agent", "session", UpdateSessionRequest(
            _PRINCIPAL, 0, "update", metadata, "work")))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            mutable["label"] = "after"
            mutable["linktools.ai.reserved"] = True
            proceed.set()
            updated = await asyncio.wait_for(task, 5)
            assert updated.metadata == {"label": "before"}
            assert updated.cwd == "work"
            replay, _ = _metadata(kind, {"label": "before"})
            assert await manager.update("agent", "session", UpdateSessionRequest(
                _PRINCIPAL, 0, "update", replay, "work")) == updated
            persisted = await state.conversation.sessions.get("session", tenant_id="tenant")
            assert persisted is not None
            assert persisted.metadata == {"label": "before"} and persisted.revision == 1
        finally:
            proceed.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("kind", ("dict", "mappingproxy", "userdict"))
@pytest.mark.parametrize("boundary", ("cwd", "authorization"))
async def test_session_create_snapshots_outer_metadata_before_validation_awaits(
    storage: PersistenceStorage, kind: str, boundary: str,
) -> None:
    entered, proceed = asyncio.Event(), asyncio.Event()

    async def canonicalize_path(value: str) -> str:
        entered.set()
        await proceed.wait()
        return value

    class Policy(TenantAuthorizationPolicy):
        async def authorize(
            self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
        ) -> None:
            await super().authorize(principal, action, resource)
            if boundary == "authorization" and action is AuthorizationAction.SESSION_CREATE:
                entered.set()
                await proceed.wait()

    async with Persistence.open("session-metadata", storage=storage, tenant_id="tenant") as state:
        manager = SessionManager(
            state.conversation, state.execution.executions, Policy(), object(),
            HmacCursorSigner("session-metadata", b"session-metadata-test-key"),
            canonicalize_path=canonicalize_path if boundary == "cwd" else None,
        )
        reserved, _ = _metadata(kind, {"linktools.ai.reserved": True})
        with pytest.raises(AIError) as invalid:
            await manager.create("agent", CreateSessionRequest(
                _PRINCIPAL, "invalid", "invalid-create", metadata=reserved))
        assert invalid.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert await state.conversation.sessions.get("invalid", tenant_id="tenant") is None
        assert await state.conversation.operations.get(
            "invalid-create", tenant_id="tenant") is None

        metadata, mutable = _metadata(kind, {"label": "before"})
        cwd = "work" if boundary == "cwd" else None
        task = asyncio.create_task(manager.create("agent", CreateSessionRequest(
            _PRINCIPAL, "session", "create", cwd=cwd, metadata=metadata)))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            mutable["label"] = "after"
            mutable["linktools.ai.reserved"] = True
            proceed.set()
            created = await asyncio.wait_for(task, 5)
            assert created.metadata == {"label": "before"}
            assert created.cwd == cwd
            receipt = await state.conversation.operations.get(
                "create", tenant_id="tenant")
            assert receipt is not None
            assert receipt.request_digest == canonical_sha256({
                "action": "session.create", "tenant_id": "tenant", "principal_id": "owner",
                "session_id": "session", "agent_id": "agent", "cwd": cwd,
                "metadata": {"label": "before"},
            })
            replay, _ = _metadata(kind, {"label": "before"})
            assert await manager.create("agent", CreateSessionRequest(
                _PRINCIPAL, "session", "create", cwd=cwd, metadata=replay)) == created
            assert await state.conversation.operations.get(
                receipt.operation_id, tenant_id="tenant") == receipt
            persisted = await state.conversation.sessions.get("session", tenant_id="tenant")
            assert persisted is not None
            assert persisted.metadata == {"label": "before"} and persisted.revision == 0
        finally:
            proceed.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
