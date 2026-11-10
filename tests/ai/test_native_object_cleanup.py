#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Object reclamation retires native metadata before its external bodies."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryImage, ModelRequest, UserPromptPart
from sqlalchemy import event, select, text

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage, SqlStepStore
from linktools.ai.persistence._archive import ContentRecord
from linktools.ai.persistence._step_schema import build_step_sql_metadata

from .test_step_content_cleanup import _Offline, _snapshot


@pytest.fixture(autouse=True)
def _exercise_offline_reclamation(monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.persistence import _native_content_index
    async def defer(_context, **kwargs):
        pass
    monkeypatch.setattr(_native_content_index, "maintain_native_content", defer)


@asynccontextmanager
async def _owners(path: Path):
    async with Persistence.open("native-gc", tenant_id="one", storage=PersistenceStorage.sqlite(path)) as state:
        context = state.execution.executions.state_store.storage_group.context
        native = SqlStepStore(context, namespace="native-gc", tenant_id="one", max_snapshots_per_run=1)
        state.run_store.bind_step_store(native)
        archive = await state.run_store.read_store(PersistenceDomain.TRACE).transcript_repository.content_archive(create=True)
        foreign = SqlStepStore(context, namespace="native-gc", tenant_id="two", max_snapshots_per_run=1)
        foreign.bind_message_archive(archive)
        try:
            yield state, native, foreign, archive
        finally:
            await foreign.close()
            await native.close()


async def _bodies(native: SqlStepStore) -> dict[tuple[str, str], ContentRecord]:
    messages = build_step_sql_metadata().tables["ai_step_messages"]
    async with native.storage_context.engine.connect() as connection:
        rows = (await connection.execute(select(messages))).mappings().all()
    return {(row["store_digest"], row["sha256"]): ContentRecord.from_native_row(row) for row in rows}


@pytest.mark.asyncio
async def test_offline_cleanup_releases_shared_content_for_online_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence import _native_content_index

    async with _owners(tmp_path / "online-resume.db") as (state, native, foreign, archive):
        await native.save_snapshot(_snapshot("run", 1, "shared", "retired"))
        latest = _snapshot("run", 2, "shared")
        await native.save_snapshot(latest)
        foreign_snapshot = _snapshot("foreign", 1, "shared", "retired")
        await foreign.save_snapshot(foreign_snapshot)
        await state.purge_unreferenced_objects((), exclusive=_Offline())
        assert await native.latest_snapshot(run_id="run") == latest
        assert await foreign.latest_snapshot(run_id="foreign") == foreign_snapshot

        monkeypatch.undo()
        await native.save_snapshot(_snapshot("run", 3))
        await _native_content_index.maintain_native_content(
            native.storage_context, store_digests=(native.store_digest,),
        )
        async with native.storage_context.engine.connect() as connection:
            for table in ("ai_step_sequences", "ai_step_messages"):
                assert await connection.scalar(text(
                    f"SELECT count(*) FROM {table} WHERE store_digest = :store"
                ), {"store": native.store_digest}) == 0
        assert await foreign.latest_snapshot(run_id="foreign") == foreign_snapshot
        await foreign.save_snapshot(_snapshot("foreign", 2))
        async with native.storage_context.engine.connect() as connection:
            for table in ("ai_step_sequences", "ai_step_messages", "ai_native_content_edges"):
                assert await connection.scalar(text(f"SELECT count(*) FROM {table}")) == 0


@pytest.mark.asyncio
async def test_object_cleanup_preserves_shared_scopes_and_republishes_retired_body(tmp_path: Path) -> None:
    async with _owners(tmp_path / "owners.db") as (state, native, foreign, archive):
        retired = replace(_snapshot("retired", 1), messages=[ModelRequest(parts=[UserPromptPart([
            "retained message", BinaryImage(b"external file", media_type="image/png"),
        ])])])
        shared = replace(retired, run_id="shared")
        await native.save_snapshot(retired)
        await native.save_snapshot(shared)
        await foreign.save_snapshot(shared)
        bodies = await _bodies(native)
        assert len(bodies) == 2
        reference, = next(iter(bodies.values())).object_refs()
        candidates = ((PersistenceDomain.TRACE, reference),)
        objects = state.archive_object_store(PersistenceDomain.TRACE)
        await native.save_snapshot(_snapshot("retired", 2))
        assert (await state.purge_unreferenced_objects(candidates, exclusive=_Offline())).retained == candidates
        assert await _bodies(native) == bodies
        await native.save_snapshot(_snapshot("shared", 2))
        assert (await state.purge_unreferenced_objects(candidates, exclusive=_Offline())).retained == candidates
        remaining = await _bodies(native)
        assert len(remaining) == 1
        assert next(iter(remaining))[0] == foreign.scope_digest(namespace="native-gc", tenant_id="two")
        assert await foreign.latest_snapshot(run_id="shared") == shared
        await foreign.save_snapshot(_snapshot("shared", 2))
        result = await state.purge_unreferenced_objects(candidates, exclusive=_Offline())
        assert result.deleted == candidates
        assert result.retained == result.missing == result.blocked == ()
        assert await _bodies(native) == {}
        async with native.storage_context.engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_sequences")) == 0
        assert await objects.stat(reference.key) is None

        revived = replace(retired, run_id="revived")
        await native.save_snapshot(revived)
        assert await objects.stat(reference.key) is not None
        reader = SqlStepStore(native.storage_context, namespace="native-gc", tenant_id="one", max_cached_messages=0)
        reader.bind_message_archive(archive)
        try:
            assert await reader.latest_snapshot(run_id="revived") == revived
        finally:
            await reader.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("rollback", "unknown", "cancelled"))
async def test_native_cleanup_failure_never_deletes_objects_and_expires_write_proofs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    async with _owners(tmp_path / "failures.db") as (state, native, foreign, archive):
        retired = replace(_snapshot("retired", 1), messages=[ModelRequest(parts=[UserPromptPart([
            "retained message", BinaryImage(b"external file", media_type="image/png"),
        ])])])
        await native.save_snapshot(retired)
        before = await _bodies(native)
        reference, = next(iter(before.values())).object_refs()
        await native.save_snapshot(_snapshot("retired", 2))
        context = native.storage_context
        async with context.engine.connect() as connection:
            edges_before = (await connection.execute(text(
                "SELECT edge_kind, source_id, target_id, revision FROM ai_native_content_edges ORDER BY id"
            ))).all()
        original = type(context).run_mutation

        async def unacknowledged(context, callback, **kwargs):
            result = await original(context, callback, **kwargs)
            if kwargs.get("domain") == "steps.content_cleanup":
                if outcome == "unknown":
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                raise asyncio.CancelledError()
            return result

        def fail_delete(connection, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith("DELETE FROM AI_STEP_MESSAGES"):
                raise RuntimeError("native body deletion rejected")

        if outcome == "rollback":
            event.listen(context.engine.sync_engine, "before_cursor_execute", fail_delete)
        else:
            monkeypatch.setattr(type(context), "run_mutation", unacknowledged)
        try:
            expected = RuntimeError if outcome == "rollback" else AIError if outcome == "unknown" else asyncio.CancelledError
            with pytest.raises(expected):
                await state.purge_unreferenced_objects(((PersistenceDomain.TRACE, reference),), exclusive=_Offline())
        finally:
            if outcome == "rollback":
                event.remove(context.engine.sync_engine, "before_cursor_execute", fail_delete)
            else:
                monkeypatch.setattr(type(context), "run_mutation", original)
        objects = state.archive_object_store(PersistenceDomain.TRACE)
        assert await objects.stat(reference.key) is not None
        assert await _bodies(native) == (before if outcome == "rollback" else {})
        async with context.engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_sequences")) == (1 if outcome == "rollback" else 0)
            edges_after = (await connection.execute(text(
                "SELECT edge_kind, source_id, target_id, revision FROM ai_native_content_edges ORDER BY id"
            ))).all()
            assert edges_after == (edges_before if outcome == "rollback" else [])
        revived = replace(retired, run_id="revived")
        await native.save_snapshot(revived)
        reader = SqlStepStore(context, namespace="native-gc", tenant_id="one", max_cached_messages=0)
        reader.bind_message_archive(archive)
        try:
            assert await reader.latest_snapshot(run_id="revived") == revived
        finally:
            await reader.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("missing", "corrupt"))
async def test_retained_attachment_damage_aborts_before_native_or_object_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    import hashlib
    from linktools.ai.storage import ObjectRef

    async with _owners(tmp_path / "damaged.db") as (state, native, _foreign, _archive):
        retained = replace(_snapshot("retained", 1), messages=[ModelRequest(parts=[UserPromptPart([
            BinaryImage(b"retained file", media_type="image/png"),
        ])])])
        await native.save_snapshot(retained)
        bodies = await _bodies(native)
        reference, = next(iter(bodies.values())).object_refs()
        objects = state.archive_object_store(PersistenceDomain.TRACE)
        orphan_data = b"orphan file"
        digest = hashlib.sha256(orphan_data).hexdigest()
        orphan = ObjectRef("runtime", reference.key.rsplit("/", 1)[0] + "/" + digest, digest, len(orphan_data))
        async def chunks():
            yield orphan_data
        await objects.put(orphan.key, chunks(), expected_size=orphan.size, expected_digest=orphan.digest)
        if damage == "missing":
            assert await objects.delete_object(reference.key, expected_digest=reference.digest)
        else:
            original = objects.open
            async def corrupt(key):
                if key == reference.key:
                    yield b"corrupt"
                else:
                    async for chunk in original(key):
                        yield chunk
            monkeypatch.setattr(objects, "open", corrupt)
        with pytest.raises(AIError):
            await state.purge_unreferenced_objects(((PersistenceDomain.TRACE, orphan),), exclusive=_Offline())
        assert await _bodies(native) == bodies
        assert await objects.stat(orphan.key) is not None
