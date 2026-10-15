# Storage caching

Caching accelerates reads; the source storage still owns durable facts. The
storage package separates content, revision discovery, and metadata projection.

| Interface | Responsibility | Host integration |
| --- | --- | --- |
| `CacheBackend` | Store discardable immutable bytes | Supply it to a domain content cache or `RuntimeStorage(..., cache_backend=...)` |
| `ContentCache[KeyT, ValueT, InfoT]` | Bind a borrowed backend to domain keys and value encoding | Supply `StorageOverlay(content_cache=...)` |
| `RevisionSource` | Observe and publish the source's current revision | Supply `StorageOverlay(revision_source=...)` |
| `StorageOverlay` | Merge layers, bind values to metadata, share content loads and refresh metadata | Keep reads through the owning store |

A host may implement these interfaces using its existing infrastructure. The
SDK does not configure Redis, create its connections, or close a host-owned
connection pool. Shared cache namespaces and deployment isolation belong to the
host adapter. Do not treat every revision as an ordered integer: revision
semantics belong to the source.

```python
from linktools.ai.asset import AssetContentCache, AssetStore
from linktools.ai.storage import StorageOverlay

store = AssetStore(StorageOverlay(
    primary=backend,
    writer=backend,
    revision_source=revision_source,
    content_cache=AssetContentCache(cache_backend),
))
await store.initialize()
try:
    content = await store.get(key)
finally:
    await store.close()
```

`ContentCache` exposes its borrowed `backend` and the domain operations
`cache_key(key, info)`, `to_cache(value)`, and `from_cache(bytes)`. The overlay
uses this one object for both identity/encoding and byte-cache access. Asset
callers use `AssetContentCache(cache_backend)`; another domain can implement the
same protocol with its own key and codec. The cache backend is never closed by
the content cache or overlay. Shared loads retain encoded bytes so each reader
can decode its own mutable value; readers do not share a mutable decoded object.

`CacheBackend` implementations provide `get`, `put`, `delete`, and
`contains_many`. `InMemoryCacheBackend`, `FilesystemCacheBackend`, and
`TieredCacheBackend` implement that contract. Empty content is a valid hit, not a
miss; it consumes at least one byte of the built-in capacity budget. These
capacity limits do not define domain retention or promise TTL expiration; the
owning domain remains responsible for its retention rules.

## Asset content digests

`AssetInfo.digest`, `AssetVersionRef.digest`, and `CapabilityLoadEntry.digest`
are the SHA-256 of the asset bytes, encoded as 64 lowercase hexadecimal
characters. Asset version payloads, SQL JSON payloads, filesystem metadata and
history, batch receipts, and snapshot manifests use the same `digest` field.
Runtime bindings preserve it through the Asset version codec. Empty assets and
non-normal entries use the SHA-256 of empty bytes.

Snapshot content keys bind to the asset digest; each manifest has its own
digest and object reference computed from the complete manifest bytes. HTTP
ETag validators are a separate transport concept.

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

Runtime storage accepts a borrowed `cache_backend` independently of asset
storage. The ordinary history and session APIs continue to own authorization,
retention and reference selection; their external object reads use the storage
router. Inline payloads do not enter the object cache. Runtime already owns its
ObjectRef/domain/tenant/source rules, so callers supply the byte backend directly;
there is no separate Runtime content adapter to construct.

```python
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.storage import InMemoryCacheBackend

cache_backend = InMemoryCacheBackend(max_bytes=16 * 1024 * 1024)
storage = RuntimeStorage.from_root("./runtime", cache_backend=cache_backend)
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

Use `TieredCacheBackend(memory, filesystem)` for local reusable content, or
`TieredCacheBackend(memory, host_cache)` when a host-provided shared backend is
useful. Existing composition can be nested for a third layer. Each additional
layer adds writes and promotion work; a filesystem cache in front of a local
filesystem source can duplicate storage without providing a useful saving.

A lower-tier hit is promoted to earlier tiers. TTL in a host backend does not
expire an already promoted memory or filesystem entry. Deletion is best effort;
source membership and domain retention remain the read authority. Host adapters
must bound network operation time as well as capacity; a failed backend is a
miss, but an unbounded network wait still delays the read.

`FilesystemCacheBackend` instances may share a directory. Mutations use one
filesystem lock and a recoverable shared usage record; local access order only
provides eviction candidates, not an independent capacity total. Each instance
calibrates on its first mutation. Missing, damaged or interrupted accounting is
rebuilt from managed regular files; unknown files are not evicted.

A successfully published `put` leaves the shared payload charge within that writer's
`max_bytes`. Budgets remain instance configuration: a smaller writer can shrink
the cache, and a larger writer can later grow it again. An inactive smaller
writer does not enforce its limit. Oversized values are skipped; `delete` removes
only its target and does not shrink unrelated entries. This is not a fixed
global-minimum quota or exact cross-process LRU.

Normal mutations inspect their target and necessary eviction candidates instead
of rescanning all content. Initial calibration, interrupted-operation recovery
and stale-candidate reconciliation have additional directory cost. Reads remain
unlocked. Cancellation waits for the mutation worker to settle before releasing
the lock. This disposable cache does not promise persistence through machine
power loss, and its payload budget is not a limit on physical disk allocation or
temporary-file peaks. All writers sharing a root must use this coordination
protocol; old implementations or external edits cannot participate safely.


## Batch reads

`BatchStorageReader.get_many(keys)` and the built-in Asset backends return an
awaitable mapping of present keys to values. Missing or deleted entries are
omitted, and repeated keys share one mapping entry.

`StorageOverlay.get_many(keys)` and `AssetStoreReader.get_many(keys)` return an
awaitable tuple in input order, including repeated keys and `None` for missing
entries. Live stores, snapshots and captured Asset readers use this facade
contract. Batch reads finish before returning; they do not expose partial
prefixes or require a stream-closing context. Origin errors propagate instead
of becoming missing values. Backend metadata batching and shared cache loaders
remain in use.
