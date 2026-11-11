# Asset storage

Asset metadata and Asset bytes have different owners. Use `SqlAssetBackend`
for persistent versioned metadata and `InMemoryAssetBackend` for temporary
versioned assets. Use `DirectoryAssetBackend` to expose an existing read-only
directory without maintaining historical versions.

`AssetStore`, `StorageOverlay` and `AssetVersionRef` retain the same logical
keys, layer identities and version-read contracts with these backends.

## Exact keys and SQL history

Asset namespaces and kinds contain at most 128 UTF-8 bytes each; opaque Asset
IDs contain at most 512 UTF-8 bytes. IDs are not paths and are never trimmed,
normalized or hashed for SQL identity. SQL uses the unique
`(namespace, kind, asset_id)` key; `asset_id` stores the exact UTF-8 bytes so
MySQL does not conflate IDs that differ by trailing spaces.

Each current entry's existing surrogate row ID also owns its history rows.
Delete and reset append revisions and retain that entry row; recreating the
same logical key retains its history. Version reads resolve the natural key
and history in one SQL statement. No identifier map or extra lookup is stored.

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
Every file, including an empty or small file, is stored in the selected
ObjectStore. Current entries, version history and batch receipts retain only
content references, digests and sizes. A filesystem ObjectStore therefore
remains useful even though the Asset metadata is in SQL. Do not remove its files
independently of the versions that still reference them.

## Directory assets are not a history database

`DirectoryAssetBackend` reads currently deployed files. Its revision read does
not reconstruct an earlier directory. AssetStore checks a captured reference's
digest and size, so modified current bytes fail the captured-version contract
rather than silently replacing the requested version.

Use a versioned backend when edits and deletions must leave earlier content
readable. Materializing a pinned version into a temporary directory does not
change the Asset's logical identity or transfer ownership to Workspace.

## Object keys and byte integrity

ObjectStore keys contain at most 1024 UTF-8 bytes across the memory, filesystem
and SQL backends. Keys stay opaque and exact, including separators, Unicode and
trailing spaces. Oversized or invalid UTF-8 keys fail before consuming content;
they are never truncated. SQL stores their original UTF-8 bytes and uniquely
indexes `(store_id, object_key)`, without a second key digest.

FilesystemObjectStore hashes the store ID and key only to derive a safe,
fixed-width physical filename. Directly using opaque keys as paths would change
separator semantics, permit path traversal and exceed filesystem filename
limits. An escaped split-path encoding could avoid that digest, but would add
path rules and directories without improving the logical identity contract.

Content SHA-256 digests bind immutable bytes to version references and detect
corruption or substitution. Batch request digests distinguish an exact replay
from a different request using the same idempotency key without retaining all
input file bytes in the receipt.

## Snapshots

`AssetStore.snapshot()` captures selected effective assets at a point in time.
It is not an export of every historical version, tombstone, reset or batch
receipt. Copying only current files or replaying ordinary `put()` calls into a
new store does not preserve arbitrary existing version references.
