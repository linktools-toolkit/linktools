#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""An existing lease selects its source from the committed admission, not a caller's layout."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, update

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._turn_sql_schema import TURN_SLOT_TABLE
from linktools.ai.storage import TurnScopeIdentity

_SCOPE = TurnScopeIdentity("turn-layout", "tenant", "runtime-pending-turn")
_FIELDS = dict(session_key="session:s", request_key="a" * 64, request_digest="b" * 64)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_coowned", (False, True))
@pytest.mark.parametrize("same_execution", (False, True))
async def test_live_replay_and_expired_takeover_across_layouts(tmp_path, first_coowned, same_execution):
    storage = PersistenceStorage.sqlite(tmp_path / "turns.db")
    async with Persistence.open(_SCOPE.namespace, tenant_id=_SCOPE.tenant_id, storage=storage) as first:
        async with Persistence.open(_SCOPE.namespace, tenant_id=_SCOPE.tenant_id, storage=storage) as second:
            first_store = first.execution.executions.state_store
            other_store = second.execution.executions.state_store
            for store in (first_store, other_store):
                store.bind_runtime_turn_scope(_SCOPE)

            first_claim = await first_store.claim_turn_admission(
                _SCOPE, **_FIELDS, execution_id="first", owner_id="a" * 32, token="b" * 32,
                duration=3600, coowned=first_coowned,
            )
            assert first_claim is not None and first_claim.lease.fence == 1

            other_mode = not first_coowned
            for route, identity in (("execution", "first"), ("session", "session:s"), ("request", "a" * 64)):
                proof = await other_store.observe_turn_admission(_SCOPE, route=route, identity=identity, coowned=other_mode)
                assert proof is not None and proof.execution_id == "first"
            assert await other_store.confirm_turn_admission(
                _SCOPE, execution_id="first", receipt="b" * 32, kind="claim", coowned=other_mode,
            ) == first_claim
            replay = await other_store.claim_turn_admission(
                _SCOPE, **_FIELDS, execution_id="proposed", owner_id="c" * 32, token="d" * 32,
                duration=3600, coowned=other_mode,
            )
            assert replay is not None and replay.execution_id == "first" and replay.lease is None
            assert await other_store.claim_turn_admission(
                _SCOPE, **{**_FIELDS, "request_key": "f" * 64}, execution_id="competing",
                owner_id="c" * 32, token="d" * 32, duration=3600, coowned=other_mode,
            ) is None
            cancelled = await other_store.change_turn_admission(
                _SCOPE, execution_id="first", receipt="e" * 32, kind="cancel", coowned=other_mode,
            )
            assert cancelled is not None and cancelled.cancel_requested
            assert await first_store.confirm_turn_admission(
                _SCOPE, execution_id="first", receipt="e" * 32, kind="cancel", coowned=first_coowned,
            ) is True

            expiry = datetime.now(timezone.utc) - timedelta(minutes=1)
            name, field = (("ai_execution_records", "turn_expiry") if first_coowned
                           else (TURN_SLOT_TABLE, "lease_expires_at"))
            table = first_store.storage_group.metadata.tables[name]
            async with first_store.context.engine.begin() as conn:
                updated = await conn.execute(update(table).where(table.c.store_key == first_store.store_key).values({field: expiry}))
                assert updated.rowcount == 1

            assert await other_store.observe_turn_admission(
                _SCOPE, route="session", identity="session:s", coowned=other_mode,
            ) is None
            execution_id = "first" if same_execution else "second"
            winner = await other_store.claim_turn_admission(
                _SCOPE, **_FIELDS, execution_id=execution_id, owner_id="c" * 32, token="d" * 32,
                duration=3600, coowned=other_mode,
            )
            assert winner is not None and winner.lease.fence == 2
            owned = await first_store.observe_turn_admission(
                _SCOPE, route="session", identity="session:s", coowned=first_coowned,
            )
            assert owned is not None and owned.execution_id == execution_id and not owned.cancel_requested
            assert await first_store.confirm_turn_admission(
                _SCOPE, execution_id="first", receipt="b" * 32, kind="claim", coowned=first_coowned,
            ) is None
            if not same_execution:
                retired = await first_store.get_turn_operation_proof(_SCOPE, "first")
                assert retired is not None and not retired.active and retired.receipt_kind == "expire"


@pytest.mark.asyncio
async def test_broken_opposite_control_link_is_not_idle(tmp_path):
    async with Persistence.open(_SCOPE.namespace, tenant_id=_SCOPE.tenant_id,
                                storage=PersistenceStorage.sqlite(tmp_path / "link.db")) as state:
        store = state.execution.executions.state_store
        store.bind_runtime_turn_scope(_SCOPE)
        assert await store.claim_turn_admission(
            _SCOPE, **_FIELDS, execution_id="first", owner_id="a" * 32, token="b" * 32,
            duration=3600, coowned=True,
        )
        table = store.storage_group.metadata.tables["ai_execution_records"]
        async with store.context.engine.begin() as conn:
            assert (await conn.execute(delete(table).where(table.c.store_key == store.store_key))).rowcount == 1
        with pytest.raises(AIError) as caught:
            await store.observe_turn_admission(_SCOPE, route="session", identity="session:s", coowned=False)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as caught:
            await store.claim_turn_admission(
                _SCOPE, **_FIELDS, execution_id="second", owner_id="c" * 32, token="d" * 32,
                duration=3600, coowned=False,
            )
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("target_coowned", (False, True))
async def test_expired_request_exchange_retires_both_actual_layouts(tmp_path, target_coowned):
    async with Persistence.open(_SCOPE.namespace, tenant_id=_SCOPE.tenant_id,
                                storage=PersistenceStorage.sqlite(tmp_path / "exchange.db")) as state:
        store = state.execution.executions.state_store
        store.bind_runtime_turn_scope(_SCOPE)
        for session, execution, request, coowned in (
            ("session:z", "target", "a" * 64, target_coowned),
            ("session:a", "incumbent", "c" * 64, not target_coowned),
        ):
            claim = await store.claim_turn_admission(
                _SCOPE, session_key=session, request_key=request, request_digest="b" * 64, execution_id=execution,
                owner_id="a" * 32, duration=3600, token="b" * 32, coowned=coowned,
            )
            assert claim is not None and claim.lease.fence == 1

        expiry = datetime.now(timezone.utc) - timedelta(minutes=1)
        async with store.context.engine.begin() as conn:
            for name, field, identity in (
                ("ai_execution_records", "turn_expiry", "turn_token"),
                (TURN_SLOT_TABLE, "lease_expires_at", "lease_token"),
            ):
                table = store.storage_group.metadata.tables[name]
                changed = await conn.execute(update(table).where(
                    table.c.store_key == store.store_key, table.c[identity].is_not(None),
                ).values({field: expiry}))
                assert changed.rowcount == 1

        received = await store.claim_turn_admission(
            _SCOPE, session_key="session:z", request_key="c" * 64, request_digest="b" * 64, execution_id="new",
            owner_id="e" * 32, duration=3600, token="f" * 32, coowned=target_coowned,
        )
        assert received is not None and received.lease.fence == 2
        assert await store.observe_turn_admission(_SCOPE, route="session", identity="session:a") is None
        for old_id in ("target", "incumbent"):
            proof = await store.get_turn_operation_proof(_SCOPE, old_id)
            assert proof is not None and not proof.active and proof.receipt_kind == "expire"


@pytest.mark.asyncio
async def test_split_to_coowned_reacquisition_does_not_rewrite_published_body(tmp_path):
    from linktools.ai.core import ExecutionStatus
    from .test_history_projection_conformance import _record

    async with Persistence.open(_SCOPE.namespace, tenant_id=_SCOPE.tenant_id,
                                storage=PersistenceStorage.sqlite(tmp_path / "published.db")) as state:
        repository = state.execution.executions
        store = repository.state_store
        store.bind_runtime_turn_scope(_SCOPE)
        record = _record(ExecutionStatus.STARTED, 1)
        await repository.prepare_content(record)
        envelope = await repository._stored_content(
            "execution", (record.execution_id,), record, state=record.status.value,
        )
        await store.mutate(lambda tx: tx.insert_record(envelope))
        assert await store.claim_turn_admission(
            _SCOPE, **_FIELDS, execution_id=record.execution_id, owner_id="a" * 32, duration=3600, token="b" * 32,
        )
        slot = store.storage_group.metadata.tables[TURN_SLOT_TABLE]
        async with store.context.engine.begin() as conn:
            await conn.execute(update(slot).where(slot.c.store_key == store.store_key).values(
                lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            ))
        after = await store.claim_turn_admission(
            _SCOPE, **_FIELDS, execution_id=record.execution_id, owner_id="c" * 32, duration=3600, token="d" * 32,
            coowned=True,
        )
        assert after is not None and after.lease.fence == 2
        assert await repository.get(record.execution_id, tenant_id=_SCOPE.tenant_id) == record
