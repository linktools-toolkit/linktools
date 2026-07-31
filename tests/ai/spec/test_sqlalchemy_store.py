import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.errors import SpecConflictError, StorageCorruptionError
from linktools.ai.spec.document import SpecDocument, SpecDocumentInfo, compute_spec_etag
from linktools.ai.spec.persistence.sqlalchemy import (
    ChangeRow,
    RevisionRow,
    SpecBlobRow,
    SqlAlchemySpecBackend,
)
from linktools.ai.storage.revision import MetadataLoadMode


def doc(path, body, *, version=1, kind="agent"):
    return SpecDocument(
        SpecDocumentInfo(path, kind, version, compute_spec_etag(body)),
        body,
    )


@pytest.mark.asyncio
async def test_sql_load_metadata_replace_then_patch_then_tombstone(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spec.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    # empty REPLACE
    load0 = await backend.load_metadata(None)
    assert load0.mode == MetadataLoadMode.REPLACE and load0.changes == ()
    await backend.put(doc("a", b"one"))
    await backend.put(doc("b", b"two"))
    # REPLACE full
    snap = await backend.load_metadata(None)
    assert snap.mode == MetadataLoadMode.REPLACE
    assert {c.key for c in snap.changes} == {"a", "b"}
    head = snap.revision
    # PATCH delta
    delta = await backend.load_metadata(0)
    assert delta.mode == MetadataLoadMode.PATCH and {c.key for c in delta.changes} == {"a", "b"}
    # empty PATCH at head
    same = await backend.load_metadata(head)
    assert same.mode == MetadataLoadMode.PATCH and same.changes == ()
    # delete -> tombstone
    await backend.delete("a")
    d2 = await backend.load_metadata(head)
    tomb = [c for c in d2.changes if c.key == "a"]
    assert len(tomb) == 1 and tomb[0].current is None
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_head_revision_matches_load_metadata_head(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spec.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    # Empty store: head is 0 (the singleton counter row is not yet seeded).
    assert await backend.head_revision() == 0
    await backend.put(doc("a", b"one"))
    # After writes, the cheap head probe matches what load_metadata would return.
    assert await backend.head_revision() == (await backend.load_metadata(None)).revision
    await backend.put(doc("b", b"two"))
    await backend.delete("a")
    assert await backend.head_revision() == (await backend.load_metadata(None)).revision
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_metadata_query_does_not_read_content(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spec.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("a", b"big-content"))
    # stat/list/load return info without reading content.
    assert (await backend.stat("a")).path == "a"
    assert [i.path for i in await backend.list_info()] == ["a"]
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_put_rejects_etag_not_matching_content(tmp_path):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    bad = SpecDocument(SpecDocumentInfo("a", "agent", 1, "not-the-sha256"), b"body")
    with pytest.raises(SpecConflictError, match="etag"):
        await backend.put(bad)
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_reset_retains_history_and_raises_minimum(tmp_path):
    # Reset raises minimum_delta_revision so old readers (after < minimum) take
    # a full REPLACE snapshot, but it no longer truncates ChangeRow -- the
    # change + version history is retained permanently for audit/rollback.
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spec.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("a", b"one"))
    await backend.put(doc("b", b"two"))
    await backend.reset((doc("b", b"two"),))  # drop a
    # Old reader (revision 0, below minimum_delta_revision) -> full REPLACE.
    after_reset = await backend.load_metadata(0)
    assert after_reset.mode == MetadataLoadMode.REPLACE
    assert {c.key for c in after_reset.changes} == {"b"}
    head = after_reset.revision
    async with backend.session_factory() as session:
        change_count = (await session.scalars(select(ChangeRow))).all()
        minimum = (await session.get(RevisionRow, 1)).minimum_delta_revision
    # History NOT cleared (the regression guard for the retention invariant).
    assert len(change_count) > 0
    assert minimum == head  # minimum raised to the reset revision
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_reset_persists_changed_and_added_content(tmp_path):
    # Regression: the reset changed/added branch must store content, not just
    # metadata (an earlier diff path built values from info-only and the NOT
    # NULL content column rejected the insert).
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spec.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("a", b"original"))
    await backend.reset(
        (
            doc("a", b"changed"),  # changed row -> UPDATE with new content
            doc("b", b"added"),  # new row -> INSERT with content
        )
    )
    assert (await backend.get("a")).content == b"changed"
    assert (await backend.get("b")).content == b"added"
    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_puts_assign_distinct_revisions(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spec.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await asyncio.gather(
        backend.put(doc("a", b"first", version=1)),
        backend.put(doc("a", b"second", version=2)),
    )
    load = await backend.load_metadata(None)
    assert load.revision == 2
    assert (await backend.get("a")).info.version == 2
    await engine.dispose()


@pytest.mark.asyncio
async def test_load_metadata_uses_one_sql_per_normal_path(tmp_path):
    # Metadata budget: first/unchanged/delta each cost exactly one SQL; only the
    # history fallback (after > head or after < minimum) may cost two. Counts
    # statements via a before_cursor_execute hook so a multi-query regression
    # cannot slip past behavioral tests.
    from sqlalchemy import event

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spec.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("a", b"one"))
    await backend.put(doc("b", b"two"))
    head = (await backend.load_metadata(None)).revision

    statements: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _catch(conn, cursor, statement, params, context, executemany):
        statements.append(statement)

    async def _select_count_for(after):
        statements.clear()
        await backend.load_metadata(after)
        return sum(1 for s in statements if s.lstrip().lower().startswith("select"))

    # First load (after=None) -> REPLACE, 1 SQL.
    assert await _select_count_for(None) == 1
    # Normal delta (0 < head) -> PATCH, 1 SQL.
    assert await _select_count_for(0) == 1
    # Unchanged (after == head) -> empty PATCH, 1 SQL.
    assert await _select_count_for(head) == 1
    # History fallback (after > head) -> REPLACE, at most 2 SQL.
    assert await _select_count_for(head + 5) <= 2
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_put_writes_blob_and_object_id(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spec.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("a", b"hello"))
    expected_sha = compute_spec_etag(b"hello")
    async with backend.session_factory() as session:
        blobs = (await session.scalars(select(SpecBlobRow))).all()
        changes = (
            await session.scalars(
                select(ChangeRow).where(ChangeRow.path == "a").order_by(ChangeRow.revision.desc())
            )
        ).all()
    assert len(blobs) == 1
    assert blobs[0].sha256 == expected_sha
    assert blobs[0].content == b"hello"
    assert changes[0].object_id == expected_sha
    assert changes[0].deleted is False
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_identical_content_dedups_blob(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spec.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("a", b"same"))
    await backend.put(doc("b", b"same"))  # same body, different path
    async with backend.session_factory() as session:
        blobs = (await session.scalars(select(SpecBlobRow))).all()
        changes_a = (
            await session.scalars(select(ChangeRow).where(ChangeRow.path == "a"))
        ).all()
        changes_b = (
            await session.scalars(select(ChangeRow).where(ChangeRow.path == "b"))
        ).all()
    assert len(blobs) == 1  # dedup: one blob for identical content
    assert changes_a[0].object_id == changes_b[0].object_id
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_delete_records_tombstone_with_null_object_id(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spec.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("a", b"x"))
    await backend.delete("a")
    async with backend.session_factory() as session:
        latest = (
            await session.scalars(
                select(ChangeRow).where(ChangeRow.path == "a").order_by(ChangeRow.revision.desc())
            )
        ).first()
        blobs = (await session.scalars(select(SpecBlobRow))).all()
    assert latest.deleted is True
    assert latest.object_id is None
    # The old content blob is retained (history preserved), not cleaned up.
    assert len(blobs) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_list_versions(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spec.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("a", b"v1"))
    await backend.put(doc("a", b"v2"))
    await backend.delete("a")
    versions = await backend.list_versions("a")
    assert len(versions) == 3
    # Newest first.
    assert [v.revision for v in versions] == sorted(
        (v.revision for v in versions), reverse=True
    )
    # The two puts carry object_ids; the tombstone does not.
    assert versions[0].deleted is True and versions[0].object_id is None
    assert versions[1].deleted is False and versions[1].object_id is not None
    assert versions[2].deleted is False and versions[2].object_id is not None
    assert versions[1].object_id != versions[2].object_id  # different content
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_get_at_revision(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spec.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("a", b"v1"))
    r1 = (await backend.load_metadata(None)).revision
    await backend.put(doc("a", b"v2"))
    r2 = (await backend.load_metadata(None)).revision
    await backend.delete("a")
    r3 = (await backend.load_metadata(None)).revision
    # Version in effect at each revision.
    assert (await backend.get_at_revision("a", r1)).content == b"v1"
    assert (await backend.get_at_revision("a", r2)).content == b"v2"
    # At the delete revision, the path is gone (tombstone).
    assert await backend.get_at_revision("a", r3) is None
    # Before the first put, the path did not exist.
    assert await backend.get_at_revision("a", r1 - 1) is None
    # A path that never existed.
    assert await backend.get_at_revision("nope", r3) is None
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_get_at_revision_missing_blob_raises_corruption(tmp_path):
    # object_id references a blob that no longer exists: every ChangeRow is
    # written in the same transaction as its blob, so this is unreachable under
    # normal operation -- when it does happen (manual tampering / data loss) the
    # backend must fail closed with StorageCorruptionError, not return None as
    # if the version never existed.
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spec.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("a", b"v1"))
    r1 = (await backend.load_metadata(None)).revision
    # Simulate data loss: delete the blob the ChangeRow points at.
    async with backend.session_factory() as session:
        async with session.begin():
            from sqlalchemy import delete as sa_delete

            await session.execute(sa_delete(SpecBlobRow))
    with pytest.raises(StorageCorruptionError, match="missing"):
        await backend.get_at_revision("a", r1)
    await engine.dispose()


# ---- apply_batch: atomic incremental batch (puts + deletes) ----------------


@pytest.mark.asyncio
async def test_apply_batch_mixed_puts_and_deletes_one_revision(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batch.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("keep", b"keep"))
    await backend.put(doc("del", b"del"))
    before = (await backend.load_metadata(None)).revision
    # One batch: put two new/update docs + delete one existing doc.
    await backend.apply_batch(
        (doc("a", b"new-a"), doc("keep", b"keep2", version=2)),
        ("del",),
    )
    after = (await backend.load_metadata(None)).revision
    assert after == before + 1, "batch must advance the revision exactly once"
    assert (await backend.get("a")).content == b"new-a"
    assert (await backend.get("keep")).content == b"keep2"
    assert await backend.get("del") is None
    await engine.dispose()


@pytest.mark.asyncio
async def test_apply_batch_insert_and_update_in_same_batch(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batch-iu.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("existing", b"old"))
    # 'existing' is an update, 'fresh' is an insert -- both in one upsert_many.
    await backend.apply_batch(
        (doc("existing", b"new", version=2), doc("fresh", b"brand-new")),
        (),
    )
    assert (await backend.get("existing")).content == b"new"
    assert (await backend.get("existing")).info.version == 2
    assert (await backend.get("fresh")).content == b"brand-new"
    await engine.dispose()


@pytest.mark.asyncio
async def test_apply_batch_delete_nonexistent_is_noop(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batch-noop.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    before = (await backend.load_metadata(None)).revision
    # Deleting a path that was never stored must not error and must not advance
    # the revision (no ChangeRow written for a no-op delete).
    await backend.apply_batch((), ("ghost",))
    after = (await backend.load_metadata(None)).revision
    assert after == before, "deleting a nonexistent path must not advance revision"
    await engine.dispose()


@pytest.mark.asyncio
async def test_apply_batch_leaves_untouched_documents_alone(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batch-untouched.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("outside", b"untouched", version=5))
    await backend.apply_batch((doc("inside", b"changed"),), ())
    # A document not mentioned in the batch is unchanged (unlike reset, which
    # would delete it).
    outside = await backend.get("outside")
    assert outside is not None
    assert outside.content == b"untouched"
    assert outside.info.version == 5
    await engine.dispose()


@pytest.mark.asyncio
async def test_apply_batch_put_overrides_delete_for_same_path(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batch-override.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("x", b"v1"))
    # Same path in both puts and deletes: the put wins (put filters its own path
    # out of the delete set), so x survives with the new content.
    await backend.apply_batch((doc("x", b"v2", version=2),), ("x",))
    result = await backend.get("x")
    assert result is not None
    assert result.content == b"v2"
    await engine.dispose()


@pytest.mark.asyncio
async def test_apply_batch_rejects_duplicate_put_paths(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batch-dup.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    with pytest.raises(SpecConflictError, match="duplicate"):
        await backend.apply_batch(
            (doc("a", b"1"), doc("a", b"2")),
            (),
        )
    await engine.dispose()


@pytest.mark.asyncio
async def test_apply_batch_empty_inputs_is_noop(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batch-empty.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    before = (await backend.load_metadata(None)).revision
    await backend.apply_batch((), ())
    after = (await backend.load_metadata(None)).revision
    assert after == before, "empty batch must not advance revision"
    await engine.dispose()


@pytest.mark.asyncio
async def test_apply_batch_records_change_history(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batch-history.db'}")
    backend = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await backend.initialize_storage(engine)
    await backend.put(doc("old", b"v1"))
    before = (await backend.load_metadata(None)).revision
    await backend.apply_batch(
        (doc("new", b"v1"),),
        ("old",),
    )
    # Probe the change log via a PATCH load (load_metadata(None) returns a
    # REPLACE snapshot of current entries only -- the 'old' tombstone is gone
    # from the current set, so query the delta since the pre-batch revision).
    load = await backend.load_metadata(before)
    changed_paths = {change.key for change in load.changes}
    assert "new" in changed_paths
    assert "old" in changed_paths
    await engine.dispose()
