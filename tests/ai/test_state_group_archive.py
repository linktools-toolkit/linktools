#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Group archive identity, transactional ownership and persisted enumeration."""

import asyncio
import hashlib
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime.state._filesystem import FilesystemStateStorageGroup, FilesystemStateStore
from linktools.ai.runtime.state._memory import InMemoryStateStore
from linktools.ai.runtime.state._sql import SqlStateStorageGroup, SqlStateStore
from linktools.ai.runtime.state._schema import build_runtime_sql_metadata
from linktools.ai.runtime.state._plan import RuntimeDomain
from linktools.ai.runtime.state._store import StateGroupTransaction, StoredRecord
from linktools.ai.storage import create_sql_storage_context

pytestmark = pytest.mark.asyncio


def _record(label: str) -> StoredRecord:
    return StoredRecord(
        hashlib.sha256(label.encode()).digest(), None, None, "probe", label,
        None, 0, None, 0, None, {"label": label},
    )


def _filesystem(root: Path, *, read_only: bool = False, standalone: bool = False) -> FilesystemStateStore:
    group = FilesystemStateStorageGroup(
        root, namespace="namespace", tenant_id="tenant", scope_digest="group",
        standalone=standalone, read_only=read_only,
    )
    return FilesystemStateStore(
        root if standalone else root / "owner", namespace="namespace", tenant_id="tenant",
        runtime_domain="owner", group=group,
    )


@pytest.mark.parametrize("backend", ("memory", "filesystem", "standalone", "sql"))
async def test_group_archive_isolated_identity_and_atomic_publication(backend: str, tmp_path: Path) -> None:
    engine = None
    if backend == "memory":
        store = InMemoryStateStore()
        other = InMemoryStateStore()
    elif backend in {"filesystem", "standalone"}:
        store = _filesystem(tmp_path / "first", standalone=backend == "standalone")
        other = _filesystem(tmp_path / "second", standalone=backend == "standalone")
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'first.db'}")
        await provision_database(engine)
        store = SqlStateStore(engine)
        other = InMemoryStateStore()
    await store.initialize()
    await other.initialize()
    try:
        group = store.storage_group
        assert await group.open_archive() is None
        scope, archive = await group.open_archive(create=True)
        assert len(scope) == 32
        assert await group.open_archive() == (scope, archive)
        assert (await other.storage_group.open_archive(create=True))[0] != scope
        owner_record, archive_record = _record("owner"), _record("content")

        async def publish(tx: StateGroupTransaction) -> None:
            await tx.transaction(store).insert_record(owner_record)
            await archive.mutate(lambda archive_tx: archive_tx.insert_record(archive_record))

        async def abort(tx: StateGroupTransaction) -> None:
            await publish(tx)
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with pytest.raises(AIError) as error:
            await group.mutate((store,), abort)
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
        assert await store.read(lambda tx: tx.get_record(owner_record.key_digest)) is None
        assert await archive.read(lambda tx: tx.get_record(archive_record.key_digest)) is None
        await group.mutate((store,), publish)
        scanned = await group.scan_stores()
        labels = []
        for partition in scanned:
            labels.extend(record.data.get("label") for record in await partition.read(lambda tx: tx.scan_records()))
        assert "owner" in labels and "content" in labels
    finally:
        await store.close()
        await other.close()
        if engine is not None:
            await engine.dispose()


@pytest.mark.parametrize("standalone", (False, True))
async def test_group_archive_filesystem_restore_readonly_and_unopened_partitions(tmp_path: Path, standalone: bool) -> None:
    root = tmp_path / "state"
    store = _filesystem(root, standalone=standalone)
    hidden = None
    if not standalone:
        hidden = FilesystemStateStore(
            root / "hidden", namespace="namespace", tenant_id="tenant",
            runtime_domain="hidden", group=store._storage_group,
        )
    await store.initialize()
    if hidden is not None:
        await hidden.initialize()
        await hidden.mutate(lambda tx: tx.insert_record(_record("hidden")))
    scope, archive = await store.storage_group.open_archive(create=True)
    await archive.mutate(lambda tx: tx.insert_record(_record("archived")))
    if hidden is not None:
        await hidden.close()
    await store.close()
    restored = tmp_path / "restored"
    shutil.copytree(root, restored)
    root = restored
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    reopened = _filesystem(root, read_only=True, standalone=standalone)
    await reopened.initialize()
    try:
        reopened_scope, archive = await reopened.storage_group.open_archive()
        assert reopened_scope == scope
        with pytest.raises(AIError) as error:
            await reopened.storage_group.open_archive(create=True)
        assert error.value.code is ErrorCode.STORAGE_READ_ONLY
        records = []
        for partition in await reopened.storage_group.scan_stores():
            records.extend(await partition.read(lambda tx: tx.scan_records()))
        assert {record.data["label"] for record in records} == ({"archived"} if standalone else {"archived", "hidden"})
    finally:
        await reopened.close()
    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert before == after


async def test_group_archive_sql_shared_identity_and_unopened_partition_scan(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.db'}")
    await provision_database(engine)
    first, second = SqlStateStore(engine), SqlStateStore(engine)
    await first.initialize()
    await second.initialize()
    try:
        results = await asyncio.gather(
            first.storage_group.open_archive(create=True),
            second.storage_group.open_archive(create=True),
        )
        assert results[0][0] == results[1][0]
        hidden = SqlStateStore(engine, group=first._storage_group, store_digest=hashlib.sha256(b"hidden").digest())
        await hidden.initialize()
        await hidden.mutate(lambda tx: tx.insert_record(_record("hidden")))
        await hidden.close()
        scanned = await second.storage_group.scan_stores()
        records = []
        for partition in scanned:
            records.extend(await partition.read(lambda tx: tx.scan_records()))
        assert any(record.data.get("label") == "hidden" for record in records)
        metadata = build_runtime_sql_metadata(frozenset({RuntimeDomain.CONVERSATION}))
        read_only = SqlStateStorageGroup(create_sql_storage_context(engine), metadata, read_only=True)
        assert (await read_only.open_archive())[0] == results[0][0]
        with pytest.raises(AIError) as error:
            await read_only.open_archive(create=True)
        assert error.value.code is ErrorCode.STORAGE_READ_ONLY
        await read_only.close()
    finally:
        await first.close()
        await second.close()
        await engine.dispose()


async def test_group_archive_filesystem_resource_scope_survives_reduced_routes(tmp_path: Path) -> None:
    from linktools.ai.runtime.state._resource_scope import StorageResourceRequest, StorageResourceScope
    from linktools.ai.storage import StorageRoute

    root = tmp_path / "state"
    route = StorageRoute.filesystem(root)
    requests = tuple(StorageResourceRequest(domain, route, root / domain) for domain in ("owner", "hidden"))
    first = await StorageResourceScope.open(requests, namespace="namespace", tenant_id="tenant")
    stores = tuple(FilesystemStateStore(
        resource.member_path, namespace="namespace", tenant_id="tenant",
        runtime_domain=request.name, group=resource.group,
    ) for request, resource in zip(requests, first.resources))
    for store in stores:
        await store.initialize()
    scope, archive = await stores[0].storage_group.open_archive(create=True)
    await archive.mutate(lambda tx: tx.insert_record(_record("archived")))
    await stores[1].mutate(lambda tx: tx.insert_record(_record("hidden")))
    for store in stores:
        await store.close()
    await first.close()
    second = await StorageResourceScope.open(requests[:1], namespace="namespace", tenant_id="tenant", read_only=True)
    resource = second.resources[0]
    store = FilesystemStateStore(resource.member_path, namespace="namespace", tenant_id="tenant", runtime_domain="owner", group=resource.group)
    await store.initialize()
    try:
        assert (await store.storage_group.open_archive())[0] == scope
        records = []
        for partition in await store.storage_group.scan_stores():
            records.extend(await partition.read(lambda tx: tx.scan_records()))
        assert {record.data["label"] for record in records} == {"archived", "hidden"}
    finally:
        await store.close()
        await second.close()


async def test_group_archive_object_resource_survives_route_reduction(tmp_path: Path) -> None:
    from linktools.ai.runtime.state._object_router import build_runtime_object_router
    from linktools.ai.runtime.state._plan import RuntimeStoragePlan
    from linktools.ai.storage import StorageRoute, read_object

    root = tmp_path / "physical"
    first = _filesystem(root)
    hidden = FilesystemStateStore(root / "hidden", namespace="namespace", tenant_id="tenant",
                                  runtime_domain="hidden", group=first.storage_group)
    plan = RuntimeStoragePlan(conversation=StorageRoute.filesystem(root / "conversation", transaction_root=root),
                              execution=StorageRoute.filesystem(root / "execution", transaction_root=root))
    router = build_runtime_object_router(plan, None, {RuntimeDomain.CONVERSATION: first, RuntimeDomain.EXECUTION: hidden}, {})
    source = router.archive_object_store(RuntimeDomain.CONVERSATION)
    assert source is router.archive_object_store(RuntimeDomain.EXECUTION)
    assert source is not router.object_store(RuntimeDomain.CONVERSATION)

    async def chunks() -> AsyncIterator[bytes]:
        yield b"archive"

    await source.put("probe", chunks(), expected_size=7, expected_digest=hashlib.sha256(b"archive").hexdigest())
    reopened = _filesystem(root, read_only=True)
    reduced = build_runtime_object_router(plan, None, {RuntimeDomain.EXECUTION: reopened}, {})
    assert await read_object(reduced.archive_object_store(RuntimeDomain.EXECUTION), "probe", expected_size=7, expected_digest=hashlib.sha256(b"archive").hexdigest()) == b"archive"
    memory = InMemoryStateStore()
    one = build_runtime_object_router(RuntimeStoragePlan(), None, {RuntimeDomain.EXECUTION: memory}, {})
    two = build_runtime_object_router(RuntimeStoragePlan(), None, {RuntimeDomain.EXECUTION: memory}, {})
    assert one.archive_object_store(RuntimeDomain.EXECUTION) is two.archive_object_store(RuntimeDomain.EXECUTION)


async def test_sql_archive_rejects_missing_identity_with_existing_content(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'scope.db'}")
    await provision_database(engine)
    owner = SqlStateStore(engine)
    await owner.initialize()
    try:
        _, archive = await owner.storage_group.open_archive(create=True)
        await archive.mutate(lambda tx: tx.insert_record(_record("content")))
        metadata = None
        for store in await owner.storage_group.scan_stores():
            if await store.read(lambda tx: tx.get_record(hashlib.sha256(b"archive_scope_id").digest())) is not None:
                metadata = store
                break
        assert metadata is not None
        await metadata.mutate(lambda tx: tx.delete_record(hashlib.sha256(b"archive_scope_id").digest()))
        for create in (False, True):
            with pytest.raises(AIError) as error:
                await owner.storage_group.open_archive(create=create)
            assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await owner.close()
        await engine.dispose()


@pytest.mark.parametrize("suffix", (".state-archive-objects", ".state-archive-objects/nested"))
async def test_group_rejects_archive_object_directory_as_state_partition(tmp_path: Path, suffix: str) -> None:
    from linktools.ai.runtime.state._resource_scope import StorageResourceRequest, StorageResourceScope
    from linktools.ai.storage import StorageRoute

    route = StorageRoute.filesystem(tmp_path, transaction_root=tmp_path)
    with pytest.raises(ValueError):
        await StorageResourceScope.open((StorageResourceRequest("owner", route, tmp_path / suffix),),
                                        namespace="namespace", tenant_id="tenant")


async def test_memory_group_owns_archive_without_borrowed_member_close_cascade() -> None:
    first = InMemoryStateStore()
    second = InMemoryStateStore(first.storage_group)
    await first.initialize()
    await second.initialize()
    group = first.storage_group
    _, archive = await group.open_archive(create=True)
    value = _record("owned-archive")
    await archive.mutate(lambda tx: tx.insert_record(value))
    await first.close()
    assert await archive.read_record(value.key_digest) == value
    assert await second.read(lambda tx: tx.scan_records()) == ()

    entered, release = asyncio.Event(), asyncio.Event()

    async def accepted_read(tx):
        entered.set()
        await release.wait()
        # Closing admission must not revoke the already-bound transaction.
        return await archive.read_record(value.key_digest)

    read = asyncio.create_task(archive.read(accepted_read))
    await entered.wait()
    closing = asyncio.create_task(group.close())
    await asyncio.sleep(0)
    assert not closing.done()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    closing = asyncio.create_task(group.close())
    with pytest.raises(AIError) as error:
        await group.open_archive()
    assert error.value.code is ErrorCode.STORAGE_CLOSED
    release.set()
    assert await read == value
    await closing
    for operation in (lambda: archive.read_record(value.key_digest), group.scan_stores):
        with pytest.raises(AIError) as error:
            await operation()
        assert error.value.code is ErrorCode.STORAGE_CLOSED
    await second.close()
    await group.close()


def _sql_archive_owner(engine) -> SqlStateStore:
    metadata = build_runtime_sql_metadata(frozenset({RuntimeDomain.CONVERSATION}))
    context = create_sql_storage_context(engine)
    group = SqlStateStorageGroup(context, metadata, uses_archive=True)
    return SqlStateStore(engine, context=context, metadata=metadata, group=group)


async def test_sql_first_owner_write_establishes_scope_in_the_same_commit(tmp_path: Path) -> None:
    from sqlalchemy import event

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'first-write.db'}")
    await provision_database(engine)
    owner = _sql_archive_owner(engine)
    await owner.initialize()
    group = owner.storage_group
    value = _record("first-owner")
    transactions = set()

    def capture(connection, cursor, statement, parameters, context, many):
        if statement.split()[0] in {"INSERT", "UPDATE", "DELETE"}:
            transactions.add(connection.get_transaction())

    async def abort(tx):
        await tx.insert_record(value)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    try:
        assert await group.open_archive() is None
        with pytest.raises(AIError):
            await owner.mutate(abort)
        assert await group.open_archive() is None
        assert await owner.read_record(value.key_digest) is None
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            await owner.mutate(lambda tx: tx.insert_record(value))
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert len(transactions) == 1 and None not in transactions
        opened = await group.open_archive()
        assert opened is not None
        assert await owner.read_record(value.key_digest) == value
    finally:
        await owner.close()
        await group.close()
        await engine.dispose()


async def test_sql_concurrent_first_owner_writes_keep_one_scope(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'concurrent-first-write.db'}"
    engines = (create_async_engine(url), create_async_engine(url))
    await provision_database(engines[0])
    stores = tuple(_sql_archive_owner(engine) for engine in engines)
    try:
        for store in stores:
            await store.initialize()
        await asyncio.gather(*(
            store.mutate(lambda tx, index=index: tx.insert_record(_record(f"owner-{index}")))
            for index, store in enumerate(stores)
        ))
        opened = await asyncio.gather(*(store.storage_group.open_archive() for store in stores))
        assert all(item is not None for item in opened)
        assert opened[0][0] == opened[1][0]
        for index in range(2):
            assert await stores[0].read_record(_record(f"owner-{index}").key_digest) == _record(f"owner-{index}")
    finally:
        for store in stores:
            await store.close()
            await store.storage_group.close()
        for engine in engines:
            await engine.dispose()


async def test_sql_owner_without_archive_declaration_does_not_create_scope(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'no-archive.db'}")
    await provision_database(engine)
    owner = SqlStateStore(engine)
    try:
        await owner.initialize()
        await owner.mutate(lambda tx: tx.insert_record(_record("plain-owner")))
        assert await owner.storage_group.open_archive() is None
    finally:
        await owner.close()
        await owner.storage_group.close()
        await engine.dispose()


async def test_sql_first_owner_write_rejects_orphan_archive_without_repair(tmp_path: Path) -> None:
    from sqlalchemy import delete

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'orphan-archive.db'}")
    await provision_database(engine)
    owner = SqlStateStore(engine)
    reopened = _sql_archive_owner(engine)
    try:
        await owner.initialize()
        _, archive = await owner.storage_group.open_archive(create=True)
        await archive.mutate(lambda tx: tx.insert_record(_record("orphan")))
        table = owner.storage_group.metadata.tables["ai_state_records"]
        async with engine.begin() as connection:
            await connection.execute(delete(table).where(
                table.c.store_digest == hashlib.sha256(b"linktools-ai:state-group-metadata").hexdigest(),
            ))
        await reopened.initialize()
        with pytest.raises(AIError) as error:
            await reopened.mutate(lambda tx: tx.insert_record(_record("rejected-owner")))
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await reopened.read_record(_record("rejected-owner").key_digest) is None
        with pytest.raises(AIError) as error:
            await reopened.storage_group.open_archive()
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        for store in (owner, reopened):
            await store.close()
            await store.storage_group.close()
        await engine.dispose()


@pytest.mark.parametrize("declared", (False, True))
@pytest.mark.parametrize("read_only", (False, True))
async def test_sql_resource_groups_aggregate_archive_owners_without_initialization_io(
    tmp_path: Path, declared: bool, read_only: bool,
) -> None:
    from linktools.ai.runtime.state._resource_scope import StorageResourceRequest, StorageResourceScope
    from linktools.ai.storage import StorageRoute

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'declared-group.db'}")
    await provision_database(engine)
    def configure(metadata):
        build_runtime_sql_metadata(frozenset({RuntimeDomain.CONVERSATION}), metadata=metadata)

    scope = await StorageResourceScope.open((
        StorageResourceRequest("ordinary", StorageRoute.sql(engine), configure_schema=configure),
        StorageResourceRequest("archive", StorageRoute.sql(engine), configure_schema=configure, uses_archive=declared),
    ), namespace="declaration", tenant_id="tenant", read_only=read_only)
    resource = scope.resources[0]
    owner = SqlStateStore(engine, context=resource.context, metadata=resource.metadata, group=resource.group)
    try:
        assert scope.resources[1].group is resource.group
        await owner.initialize()
        assert await resource.group.open_archive() is None
        if read_only:
            with pytest.raises(AIError) as error:
                await owner.mutate(lambda tx: tx.insert_record(_record("owner")))
            assert error.value.code is ErrorCode.STORAGE_READ_ONLY
        else:
            await owner.mutate(lambda tx: tx.insert_record(_record("owner")))
        assert (await resource.group.open_archive() is not None) is (declared and not read_only)
    finally:
        await owner.close()
        await scope.close()
        await engine.dispose()


@pytest.mark.parametrize("committed", (False, True))
async def test_sql_first_archive_owner_unknown_commit_keeps_scope_with_owner_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, committed: bool,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'unknown-first-write.db'}")
    await provision_database(engine)
    owner = _sql_archive_owner(engine)
    value = _record("uncertain-owner")
    group = owner.storage_group
    original = type(group.context).run_mutation

    async def unknown(context, callback, **kwargs):
        if context is group.context:
            if committed:
                await original(context, callback, **kwargs)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        return await original(context, callback, **kwargs)

    try:
        await owner.initialize()
        with monkeypatch.context() as patch:
            patch.setattr(type(group.context), "run_mutation", unknown)
            with pytest.raises(AIError) as error:
                await owner.mutate(lambda tx: tx.insert_record(value))
            assert error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert await owner.read_record(value.key_digest) == (value if committed else None)
        opened = await group.open_archive()
        assert (opened is not None) is committed
        await owner.mutate(lambda tx: tx.insert_record(_record("later-owner")))
        after = await group.open_archive()
        assert after is not None
        if committed:
            assert after[0] == opened[0]
    finally:
        await owner.close()
        await group.close()
        await engine.dispose()
