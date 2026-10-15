# Storage caching

Caching accelerates reads; the source storage still owns durable facts. The
storage package separates content, revision discovery, and metadata projection.

| Interface | Responsibility | Host integration |
| --- | --- | --- |
| `ContentCache` | Store discardable immutable byte content | Supply a cache implementation through `StorageOverlay(cache=...)` |
| `RevisionSource` | Observe and publish the source's current revision | Supply `StorageOverlay(revision_source=...)` |
| `StorageOverlay` | Merge layers, bind values to metadata, share content loads and refresh metadata | Keep reads through the owning store |

A host may implement these interfaces using its existing infrastructure. The
SDK does not configure Redis, create its connections, or close a host-owned
connection pool. Shared cache namespaces and deployment isolation belong to the
host adapter. Do not treat every revision as an ordered integer: revision
semantics belong to the source.

```python
from linktools.ai.asset import AssetCacheAdapter, AssetStore
from linktools.ai.storage import StorageOverlay

store = AssetStore(StorageOverlay(
    primary=backend,
    writer=backend,
    revision_source=revision_source,
    cache=content_cache,
    cache_adapter=AssetCacheAdapter(),
))
await store.initialize()
try:
    content = await store.get(key)
finally:
    await store.close()
```

`ContentCache` implementations provide `get`, `put`, `delete`, and
`contains_many`. `InMemoryContentCache`, `FilesystemContentCache`, and
`TieredContentCache` implement that contract. Empty content is a valid hit, not a
miss; it consumes at least one byte of the built-in capacity budget. These
capacity limits do not define domain retention or promise TTL expiration; the
owning domain remains responsible for its retention rules.

## Freshness and cancellation

A content key must identify the exact content version. Reading a revision and
then caching unrelated latest content under it is not a versioned read.
Metadata readers entering before a shared source observation begins may reuse
that observation. A later reader must still observe writes committed before its
read; request coalescing must not silently weaken freshness.

An observer does not own a shared load. Cancelling one observer does not cancel
other observers' source work. The owning storage lifecycle settles its work
before closing its resources. Request-owned local work has a different
cancellation contract and must not be moved into a shared background load.

`StorageOverlay.invalidate()` discards materialized metadata after an external
mutation. It is not a global cache flush or a promise of cross-process
read-after-write consistency. `refresh()` follows the configured revision
source and layer policies; it is not a universal forced-origin read.

## Boundaries

Runtime authorization, leases, cancellation, commit confirmation and recovery
remain authoritative operations. Cache hits do not grant access. A cached
immutable value must not remain accessible after its owning domain has deleted
it or expired its retention scope. Transient or volatile content must not gain
persistent filesystem or remote copies merely because its bytes are immutable.

Live event delivery and reconnect cursors are separate from content caching.
A shared byte cache does not provide event ordering, subscription recovery, or
producer fencing. Likewise, business TTL snapshots and explicit refresh
semantics remain with their read owner unless an equivalent contract has been
established. Existing Runtime history calls do not automatically acquire shared
content caching from the asset-store configuration above.
