# Storage caching

Caching accelerates reads; the source storage still owns durable facts. The
storage package separates content, revision discovery, and metadata projection.

| Interface | Responsibility | Host integration |
| --- | --- | --- |
| `ContentCache` | Store discardable immutable byte content | Supply `StorageOverlay(cache=...)` or `RuntimeStorage(..., content_cache=...)` |
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

## Runtime content

Runtime storage accepts a borrowed `content_cache` independently of asset
storage. The ordinary history and session APIs continue to own authorization,
retention and reference selection; their external object reads use the storage
router. Inline payloads do not enter the object cache.

```python
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.storage import InMemoryContentCache

content_cache = InMemoryContentCache(max_bytes=16 * 1024 * 1024)
storage = RuntimeStorage.from_root("./runtime", content_cache=content_cache)
```

The host can retain the cache across successive storage lifetimes. Different
storage instances share cached bytes, not their in-flight loads. The owning
storage settles its own reads on close without closing the borrowed cache or a
host connection pool. Closing first completes pending archive projections,
then drains accepted object reads before source storage is closed.

Only durable routes use this cache. A cold read uses the source directly. A
cache hit verifies the expected size and digest and checks the referenced
object still exists in the current source. Deletion or a source error cannot
be hidden by cached bytes. This saves body transfer and filesystem work; it
does not eliminate authorization, state, reference or object-metadata queries.
`RuntimeStorage.read_object(domain, reference)` is the storage-level read
operation used by SDK composition; applications normally keep using their
existing domain/history APIs.

## Choosing layers

Use `TieredContentCache(memory, filesystem)` for local reusable content, or
`TieredContentCache(memory, host_cache)` when a host-provided shared backend is
useful. Existing composition can be nested for a third layer. Each additional
layer adds writes and promotion work; a filesystem cache in front of a local
filesystem source can duplicate storage without providing a useful saving.

A lower-tier hit is promoted to earlier tiers. TTL in a host backend does not
expire an already promoted memory or filesystem entry. Deletion is best effort;
source membership and domain retention remain the read authority. Host adapters
must bound network operation time as well as capacity; a failed backend is a
miss, but an unbounded network wait still delays the read.

One active `FilesystemContentCache` instance must own a cache directory. Its
index and capacity accounting are instance-local. Sharing that directory
between independent instances or worker processes does not provide a shared
LRU or a global capacity bound. Hosts can use separate stable worker directories
and account for the sum of their budgets and abandoned-directory cleanup.
