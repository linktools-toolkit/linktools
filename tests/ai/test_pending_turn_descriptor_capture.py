#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conversation descriptor capture preserves owner identity and fresh publication."""
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeDomain, RuntimeState
from linktools.ai.runtime.state._conversation_repositories import ConversationHistoryRepositoryImpl
from linktools.ai.runtime.state._store import active_state_scope
from .test_incremental_projection_boundaries import _storage
from .test_pending_turn_commit import (
    _NAMESPACE, _commit, _pending_turn,
    _snapshot, _turn_state,
)

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("damage", ("missing", "wrong_session", "wrong_history_id"))
async def test_descriptor_failure_never_publishes_owners(tmp_path: Path, damage: str) -> None:
    async with _turn_state("sqlite", tmp_path) as state:
        owner = state.storage.conversation.histories
        key = owner.history_record_key("history")
        async def damage_record(tx):
            stored = await tx.get_record(key)
            if damage == "missing":
                await tx.delete_record(key)
            else:
                descriptor = await owner.get_in_transaction(tx, "history", tenant_id="tenant")
                replacement = replace(descriptor, **{
                    "session_id" if damage == "wrong_session" else "history_id": "another-identity",
                })
                raw = owner._stored("conversation_history", "history", replacement)
                assert await tx.replace_record(replace(raw, storage_revision=stored.storage_revision + 1),
                                               expected_storage_revision=stored.storage_revision)
        await owner.state_store.mutate(damage_record)
        before = await _snapshot(state)
        with pytest.raises(AIError) as raised:
            await _commit(state, _pending_turn(state))
        assert raised.value.code in {ErrorCode.STORAGE_INTEGRITY_ERROR, ErrorCode.STORAGE_CONFLICT}
        assert await _snapshot(state) == before
        assert active_state_scope() is None


@pytest.mark.parametrize("damage", ("session_identity", "branch_identity"))
async def test_post_capture_descriptor_change_is_rejected_by_fresh_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    async with _turn_state("sqlite", tmp_path) as state:
        owner = state.storage.conversation.histories
        archive = state.storage.run_store.read_store(RuntimeDomain.CONVERSATION)
        original = archive.prepare_conversation_checkpoint
        snapshots = []
        async def changed_after_capture(run, checkpoint, **kwargs):
            result = await original(run, checkpoint, **kwargs)
            assert active_state_scope() is None
            assert kwargs.get("observed_records") is not None
            captured = await owner.get("history", tenant_id="tenant")
            assert captured.history_id == "history"
            key = owner.history_record_key("history")
            async def replace_descriptor(tx):
                old = await tx.get_record(key)
                replacement = replace(captured, **{
                    "session_id" if damage == "session_identity" else "history_id": "changed-after-capture",
                })
                new = owner._stored("conversation_history", "history", replacement)
                assert await tx.replace_record(replace(new, storage_revision=old.storage_revision + 1),
                                               expected_storage_revision=old.storage_revision)
            await owner.state_store.mutate(replace_descriptor)
            snapshots.append(await _snapshot(state))
            return result
        monkeypatch.setattr(archive, "prepare_conversation_checkpoint", changed_after_capture)
        with pytest.raises(AIError) as raised:
            await _commit(state, _pending_turn(state))
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert len(snapshots) == 1 and await _snapshot(state) == snapshots[0]


@pytest.mark.parametrize("scope", ("namespace", "tenant", "other_group", "custom_owner"))
async def test_unmatched_owner_keeps_original_read_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str,
) -> None:
    async with _turn_state("sqlite", tmp_path / "first") as state:
        actual = state.storage.conversation.histories
        other = None
        calls = []
        if scope == "other_group":
            other = RuntimeState(_storage("sqlite", tmp_path / "second"))
            await other.initialize(namespace=_NAMESPACE, tenant_id="tenant")
            await other.conversation.sessions.create(state.session)
            owner = other.conversation.histories
        elif scope == "custom_owner":
            class CustomOwner(ConversationHistoryRepositoryImpl):
                pass
            owner = CustomOwner(actual.state_store, namespace=_NAMESPACE, tenant_id="tenant")
        else:
            owner = ConversationHistoryRepositoryImpl(
                actual.state_store,
                namespace="foreign" if scope == "namespace" else _NAMESPACE,
                tenant_id="foreign" if scope == "tenant" else "tenant",
            )
        original = owner.get
        async def standalone(*args, **kwargs):
            assert active_state_scope() is None
            calls.append((args, kwargs))
            return await original(*args, **kwargs)
        monkeypatch.setattr(owner, "get", standalone)
        monkeypatch.setattr(state.commands, "_conversation_history", owner)
        try:
            if scope in {"namespace", "tenant"}:
                before = await _snapshot(state)
                with pytest.raises(AIError) as raised:
                    await _commit(state, _pending_turn(state))
                assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
                assert await _snapshot(state) == before
            else:
                result = await _commit(state, _pending_turn(state))
                assert result.execution.execution_id == "execution"
            assert len(calls) == 1
        finally:
            if other is not None:
                await other.close()


@pytest.mark.parametrize("identity", ("same_request", "different_request", "conflicting_digest"))
async def test_late_winner_after_capture_remains_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: str,
) -> None:
    async with _turn_state("sqlite", tmp_path) as state:
        archive = state.storage.run_store.read_store(RuntimeDomain.CONVERSATION)
        original = archive.prepare_conversation_checkpoint
        late = _pending_turn(state, execution_id="late-winner", answer="winner")
        contender = _pending_turn(
            state, execution_id="contender", answer="loser",
            key="c" * 64 if identity == "different_request" else "a" * 64,
            request_digest="d" * 64 if identity == "conflicting_digest" else "b" * 64,
        )
        winners = []
        snapshots = []
        injecting = False
        async def win_after_capture(run, checkpoint, **kwargs):
            nonlocal injecting
            result = await original(run, checkpoint, **kwargs)
            if not injecting:
                assert kwargs.get("observed_records") is not None
                injecting = True
                winners.append(await _commit(state, late))
                snapshots.append(await _snapshot(state))
            return result
        monkeypatch.setattr(archive, "prepare_conversation_checkpoint", win_after_capture)
        if identity == "same_request":
            assert await _commit(state, contender) == winners[0]
        else:
            with pytest.raises(AIError) as raised:
                await _commit(state, contender)
            assert raised.value.code in {
                ErrorCode.SESSION_BUSY, ErrorCode.STORAGE_CONFLICT, ErrorCode.IDEMPOTENCY_CONFLICT,
            }
        assert len(winners) == 1 and await _snapshot(state) == snapshots[0]
        assert await state.storage.execution.executions.get("contender", tenant_id="tenant") is None
        assert active_state_scope() is None


@pytest.mark.parametrize("backend", ("memory", "filesystem"))
async def test_unsupported_group_keeps_descriptor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        owner = state.storage.conversation.histories
        calls = []
        original = owner.get
        async def read(*args, **kwargs):
            calls.append(True)
            return await original(*args, **kwargs)
        monkeypatch.setattr(owner, "get", read)
        result = await _commit(state, _pending_turn(state))
        assert result.execution.execution_id == "execution"
        assert calls
