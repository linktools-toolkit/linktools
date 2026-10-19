#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unknown commits can be confirmed later without attempting another write."""

import asyncio

import pytest

from linktools.ai.runtime.state import RuntimeState
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._contracts import RuntimeDomain, RuntimePayloadRef
from linktools.ai.storage import StoredPayload
from linktools.ai.runtime.state._store import active_state_scope

from tests.ai.test_pending_turn_commit import _NAMESPACE, _TurnState, _commit, _pending_turn, _runtime_commands, _snapshot, _storage, _turn_state
from tests.ai.test_pending_turn_sessionless import _commit as _commit_standalone, _standalone

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("standalone", (False, True))
async def test_terminal_unknown_can_later_confirm_source_without_mutation(tmp_path, monkeypatch, backend, standalone):
    async with _turn_state(backend, tmp_path) as state:
        turn = _standalone(state) if standalone else _pending_turn(state)
        expected = None if standalone else state.session
        group = state.storage.execution.executions.state_store.storage_group
        mutate, read_many = type(group).mutate, type(group).read_many
        lost = False

        async def lost_ack(self, stores, callback):
            nonlocal lost
            result = await mutate(self, stores, callback)
            if self is group:
                lost = True
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        async def unavailable(self, stores, callback):
            if self is group and lost:
                raise RuntimeError("source temporarily unavailable")
            return await read_many(self, stores, callback)

        assert await state.commands.read_pending_turn(turn.reservation, expected_session=expected) is None
        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", lost_ack)
            patch.setattr(type(group), "read_many", unavailable)
            with pytest.raises(AIError) as raised:
                await (_commit_standalone(state, turn) if standalone else _commit(state, turn))
            assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        before = await _snapshot(state)

        async def forbidden(*args, **kwargs):
            raise AssertionError("source confirmation must not mutate")

        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", forbidden)
            result = await state.commands.read_pending_turn(turn.reservation, expected_session=expected)
            assert result is not None and result.result == turn.commit.result
            assert await state.commands.read_pending_turn(turn.reservation, expected_session=expected) == result
            if expected is not None:
                current = await state.storage.conversation.sessions.get(expected.session_id, tenant_id="tenant")
                assert await state.commands.read_pending_turn(turn.reservation, expected_session=current) == result
        assert await _snapshot(state) == before


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("standalone", (False, True))
@pytest.mark.parametrize("prepared", (False, True))
async def test_promotion_unknown_confirms_only_original_source_receipt(tmp_path, monkeypatch, backend, standalone, prepared):
    async with _turn_state(backend, tmp_path) as state:
        turn = _standalone(state) if standalone else _pending_turn(state)
        expected = None if standalone else state.session
        context = RuntimePayloadRef(StoredPayload.inline_bytes(b"[]"), RuntimeDomain.EXECUTION) if prepared else None
        claim = "original-producer"
        group = state.storage.execution.executions.state_store.storage_group
        mutate, read_many = type(group).mutate, type(group).read_many
        lost = False
        inside = False

        async def lost_ack(self, stores, callback):
            nonlocal lost, inside
            if self is not group or inside:
                return await mutate(self, stores, callback)
            inside = True
            try:
                await mutate(self, stores, callback)
            finally:
                inside = False
            lost = True
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        async def unavailable(self, stores, callback):
            if self is group and lost:
                raise RuntimeError("source temporarily unavailable")
            return await read_many(self, stores, callback)

        assert await state.commands.read_pending_turn_promotion(
            turn.reservation, expected_session=expected,
            captured_input_context=context, producer_claim_id=claim,
        ) is None
        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", lost_ack)
            patch.setattr(type(group), "read_many", unavailable)
            with pytest.raises(AIError) as raised:
                await state.commands.promote_pending_turn(
                    turn.reservation, expected_session=expected,
                    captured_input_context=context, producer_claim_id=claim,
                )
            assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        before = await _snapshot(state)

        async def forbidden(*args, **kwargs):
            raise AssertionError("source confirmation must not mutate")

        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", forbidden)
            result = await state.commands.read_pending_turn_promotion(
                turn.reservation, expected_session=expected,
                captured_input_context=context, producer_claim_id=claim,
            )
            assert result is not None and result[0].revision == (3 if prepared else 1)
            if prepared:
                with pytest.raises(AIError) as raised:
                    await state.commands.read_pending_turn_promotion(
                        turn.reservation, expected_session=expected,
                        captured_input_context=context, producer_claim_id="foreign",
                    )
                assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert await _snapshot(state) == before

@pytest.mark.parametrize("prepared", (False, True))
async def test_promotion_concurrent_conflicting_digest_is_definite_conflict(tmp_path, monkeypatch, prepared):
    async with _turn_state("sqlite", tmp_path) as first:
        other = RuntimeState(_storage("sqlite", tmp_path))
        await other.initialize(namespace=_NAMESPACE, tenant_id="tenant")
        try:
            second = _TurnState(other, _runtime_commands(other, _NAMESPACE), first.session)
            turns = (_pending_turn(first, execution_id="first"),
                     _pending_turn(second, execution_id="second", request_digest="d" * 64))
            groups = tuple(state.storage.execution.executions.state_store.storage_group for state in (first, second))
            original = type(groups[0]).mutate
            entered, ready = set(), asyncio.Event()

            async def mutate(self, stores, callback):
                if self in groups and active_state_scope() is None and self not in entered:
                    entered.add(self)
                    if len(entered) == 2:
                        ready.set()
                    await asyncio.wait_for(ready.wait(), 5)
                return await original(self, stores, callback)

            context = RuntimePayloadRef(StoredPayload.inline_bytes(b"[]"), RuntimeDomain.EXECUTION) if prepared else None
            with monkeypatch.context() as patch:
                patch.setattr(type(groups[0]), "mutate", mutate)
                results = await asyncio.wait_for(asyncio.gather(*(
                    state.commands.promote_pending_turn(
                        turn.reservation, expected_session=state.session, captured_input_context=context,
                        producer_claim_id="claim-" + turn.reservation.execution.execution_id,
                    ) for state, turn in zip((first, second), turns)
                ), return_exceptions=True), 15)
            assert len([value for value in results if isinstance(value, tuple)]) == 1, results
            errors = [value for value in results if isinstance(value, AIError)]
            assert len(errors) == 1 and errors[0].code in {ErrorCode.IDEMPOTENCY_CONFLICT, ErrorCode.SESSION_BUSY}, results
        finally:
            await other.close()
