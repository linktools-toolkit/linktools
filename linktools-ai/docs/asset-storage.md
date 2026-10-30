# Asset storage

Asset metadata and Asset bytes have different owners. Use `SqlAssetBackend`
for persistent versioned metadata and `InMemoryAssetBackend` for temporary
versioned assets. Use `DirectoryAssetBackend` to expose an existing read-only
directory without maintaining historical versions.

`AssetStore`, `StorageOverlay` and `AssetVersionRef` retain the same logical
keys, layer identities and version-read contracts with these backends.

## Local persistent assets

Install the `sqlite` extra for local SQL storage. Provision the schema explicitly;
opening an AssetStore does not perform deployment migrations.

```python
from pathlib import Path

from sqlalchemy import URL
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.asset import AssetKey, AssetStore, SqlAssetBackend
from linktools.ai.migrate import provision_asset_database
from linktools.ai.storage import FilesystemObjectStore, StorageOverlay

root = Path("asset-data")
root.mkdir(parents=True, exist_ok=True)
engine = create_async_engine(
    URL.create("sqlite+aiosqlite", database=str(root / "assets.db"))
)
try:
    # Run this at the explicit provisioning boundary, not on every request.
    await provision_asset_database(engine)
    backend = SqlAssetBackend(
        engine,
        namespace="application-assets",
        object_store=FilesystemObjectStore(root / "objects"),
    )
    store = AssetStore(StorageOverlay(backend, writer=backend))
    try:
        await store.initialize()
        key = AssetKey("resource", "sample")
        await store.put(key, b"first")
        pinned = (await store.resolve_versions((key,)))[0]
        await store.put(key, b"second")
        assert await store.read_versions((pinned,)) == (b"first",)
    finally:
        await store.close()
finally:
    await engine.dispose()
```

The application owns the engine and injected object store. Close active Asset
users before disposing those resources. For a shared SQL database, pass its
existing provisioned engine instead of creating a local engine.

Without an injected ObjectStore, the SQL backend uses `SqlObjectStore`.
`PayloadPolicy` keeps small content inline and places larger content in the
selected ObjectStore. A filesystem ObjectStore therefore remains useful even
though the Asset metadata is in SQL. Do not remove its files independently of
the versions that still reference them.

## Directory assets are not a history database

`DirectoryAssetBackend` reads currently deployed files. Its revision read does
not reconstruct an earlier directory. AssetStore checks a captured reference's
digest and size, so modified current bytes fail the captured-version contract
rather than silently replacing the requested version.

Use a versioned backend when edits and deletions must leave earlier content
readable. Materializing a pinned version into a temporary directory does not
change the Asset's logical identity or transfer ownership to Workspace.

## Replacing a versioned filesystem backend

The `FilesystemAssetBackend` and `filesystem_root` APIs are removed. They are
not aliases for SQLite or DirectoryAssetBackend. This source change does not
convert or delete an existing metadata directory, create a substitute database,
or rewrite a stored AssetVersionRef.

Before upgrading a consumer that used the removed backend:

1. Inventory its actual backend, roots, object stores and any executions or
   retained references that still need old versions. A directory path alone
   does not identify the backend.
2. Keep a quiescent copy of the original metadata and objects, with a reader
   capable of opening that format. Do not run mixed old and new writers against
   the same data.
3. If historical state must remain live, complete and verify an explicit data
   conversion before switching the consumer. Preserve logical keys, layer IDs,
   entry revisions, store revision ordering, metadata, deletion/reset states,
   version bytes and idempotency receipts. Reopen and read the references that
   existing consumers actually retain.
4. Switch to a separately provisioned target only after those checks. Retain
   the source until its independent data-retention requirements are resolved.

There is no automatic historical converter in this change. Consumers with live
filesystem history must retain their compatible reader until a conversion is
available; an empty SQL store is not a successful migration.

`AssetStore.snapshot()` captures selected effective assets at a point in time.
It is not an export of every historical version, tombstone, reset or batch
receipt. Copying only current files or replaying ordinary `put()` calls into a
new store does not preserve arbitrary existing version references.

Existing SQL metadata, ObjectStore implementations and Directory readers are
not reinterpreted by this backend removal. Runtime's
`PersistenceStorage.from_root()` is a separate composition: it already uses
SQLite state and filesystem objects, not this Asset metadata format.

