#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session history reads the canonical terminal Execution without loading its binding."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._snapshot_validation import validate_snapshot_references
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore

from .test_session_timeline import _execution, _session, _commit_empty_turn
from .test_session_timeline_window import _window_host


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_timeline_projection_reads_terminal_owners_without_expanding_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("timeline-owner-projection", storage=storage, tenant_id="tenant") as persistence:
        repository = persistence.execution.executions
        values = tuple(_execution(str(index), status=status, prompt=str(index)) for index, status in enumerate((
            ExecutionStatus.STARTED, ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED,
        )))
        for value in values:
            await repository.create(value)

        async def no_binding(*_args, **_kwargs):
            pytest.fail("timeline reads must not materialize executable bindings")

        monkeypatch.setattr(repository._execution_content, "expand", no_binding)
        monkeypatch.setattr(repository._execution_content, "expand_many", no_binding)
        monkeypatch.setattr(repository._execution_content, "expand_bindings", no_binding)
        projected = await repository.get_timeline_many(tuple(value.execution_id for value in values), tenant_id="tenant")
        assert set(projected) == {value.execution_id for value in values}
        for value in values:
            record = projected[value.execution_id]
            assert record.result == value.result and record.stored_user_input == value.stored_user_input
            assert record.completed_at == value.completed_at
        with pytest.raises(AIError) as denied:
            await repository.get_timeline_many(("0",), tenant_id="other")
        assert denied.value.code is ErrorCode.STORAGE_OWNER_MISMATCH


def test_terminal_completion_time_is_independent_of_control_updates() -> None:
    execution = _execution("execution", status=ExecutionStatus.SUCCEEDED, prompt="original")
    updated = replace(execution, updated_at=execution.updated_at + timedelta(days=1),
                      revision=execution.revision + 1, retention_closed=True)
    assert updated.finalize_terminal(updated.result).completed_at == execution.completed_at


@pytest.mark.parametrize("status", (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED))
def test_timeline_projection_preserves_unfinalized_owner_state(status: ExecutionStatus) -> None:
    from linktools.ai.persistence.codec import _encode_persisted_domain, decode_execution_timeline, encode_envelope

    execution = replace(_execution("execution", status=status, prompt="input"), result=None, completed_at=None)
    projected = decode_execution_timeline(encode_envelope({
        "type": "execution_record", "payload": _encode_persisted_domain(execution),
    }))
    assert projected.status is status
    assert projected.result is None and projected.completed_at is None


def test_timeline_projection_inherits_execution_constructor_defaults() -> None:
    from linktools.ai.persistence.codec import _encode_persisted_domain, decode_execution_timeline, encode_envelope

    execution = _execution("execution", status=ExecutionStatus.STARTED, prompt="input")
    payload = _encode_persisted_domain(execution)
    for name in ("result", "error_diagnostics", "completed_at", "conversation"):
        del payload["fields"][name]
    projected = decode_execution_timeline(encode_envelope({"type": "execution_record", "payload": payload}))
    assert projected.result is None and projected.error_diagnostics is None
    assert projected.completed_at is None and projected.conversation is True


@pytest.mark.parametrize("malformed", ("missing_identity", "explicit_invalid", "schema"))
def test_timeline_projection_rejects_invalid_explicit_values(malformed: str) -> None:
    from linktools.ai.persistence.codec import _encode_persisted_domain, decode_execution_timeline, encode_envelope

    execution = _execution("execution", status=ExecutionStatus.STARTED, prompt="input")
    payload = _encode_persisted_domain(execution)
    if malformed == "missing_identity":
        del payload["fields"]["execution_id"]
    elif malformed == "explicit_invalid":
        payload["fields"]["conversation"] = None
    else:
        payload["schema"] = True
    with pytest.raises(AIError):
        decode_execution_timeline(encode_envelope({"type": "execution_record", "payload": payload}))


@pytest.mark.asyncio
async def test_snapshot_rejects_a_memory_turn_without_its_canonical_execution() -> None:
    async with Persistence.open("timeline-snapshot-owner", storage=PersistenceStorage.in_memory(),
                                tenant_id="tenant") as state:
        await state.conversation.sessions.create(_session())
        value = _execution("execution", status=ExecutionStatus.FAILED, prompt="original")
        await state.execution.executions.create(value)
        await state.conversation.sessions.admit_execution("session", tenant_id="tenant", execution_id=value.execution_id, expected=None)
        await _commit_empty_turn(state, value)
        records = {
            domain: await store.read(lambda tx: tx.scan_records())
            for domain, store in ((PersistenceDomain.EXECUTION, state.execution.executions.state_store),
                                  (PersistenceDomain.CONVERSATION, state.conversation.sessions.state_store))
        }
        validate_snapshot_references(records, {})
        records[PersistenceDomain.EXECUTION] = ()
        with pytest.raises(AIError) as missing:
            validate_snapshot_references(records, {})
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_native_snapshot_rejects_a_turn_without_its_session_owner(tmp_path: Path) -> None:
    archive = InMemoryObjectStore("native-session-owner")
    limits = SnapshotLimits(max_entries=1000, max_bytes=4 * 1024 * 1024)
    async def export():
        async with Persistence.open(
            "window", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
            tenant_id="tenant", read_only=True,
        ) as reader:
            return await reader.export_snapshot(object_store=archive, limits=limits)

    async with _window_host(tmp_path) as (state, runtime, principal, _, _):
        turn = await runtime.sessions.start(
            "session", "original", agent="default", principal=principal, idempotency_key="capture",
        )
        assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        assert await export()

        store = state.session_turns
        async with store._context.engine.begin() as connection:
            removed = await connection.execute(store._sessions.delete().filter_by(**store._session_values("session")))
            assert removed.rowcount == 1

        with pytest.raises(AIError) as missing:
            await export()
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
