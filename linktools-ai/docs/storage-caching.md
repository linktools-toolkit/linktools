# Storage caching

Caching accelerates reads; the source storage still owns durable facts. The
storage package separates content, revision discovery, and metadata projection.

`RuntimeStorage` declares reusable resource routes and an optional object store.
Each Runtime binding acquires and closes its own operational scope; borrowed
backends remain host-owned. Stores
such as `AssetStore`, `ObjectStore`, and `MetricStore` expose their respective
data contracts; repositories own domain records and state transitions. Backends
provide replaceable source or transport capabilities. These roles are not
determined by whether a type is a protocol or a concrete implementation, and a
store does not necessarily imply durable retention.

| Interface | Responsibility | Host integration |
| --- | --- | --- |
| `CacheBackend` | Store discardable immutable bytes | Supply it to a domain content cache or `RuntimeStorage.sqlite(path, cache=...)` |
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

`CacheBackend` implementations provide `get`, `put`, and `delete`;
`contains_many` is an optional batch membership optimization. `InMemoryCacheBackend`, `FilesystemCacheBackend`, and
`TieredCacheBackend` implement that contract. Empty content is a valid hit, not a
miss; it consumes at least one byte of the built-in capacity budget. These
capacity limits do not define domain retention or promise TTL expiration; the
owning domain remains responsible for its retention rules.

## Read resources and write policies

`StorageRoute(..., cache=...)` and the RuntimeStorage factories borrow a
`CacheBackend` for hot reads only. `CommitPolicy.batch(buffer=..., ...)` borrows
an independent `WriteBufferBackend` for pending writes. The same implementation
may provide both capabilities, but each use is explicitly configured.
`ExecutionCapability(coordination=...)` separately supplies turn ownership and
live streams, including the non-authoritative pending-turn view.

Adapters implement complete method groups: `get`/`put`/`delete` for hot content,
`append`/`read_batch`/`ack` for pending writes, and `open_stream`/`read_stream`
for previews. A write-only adapter is valid as a policy buffer, not as a
StorageRoute cache. Pending bytes must not inherit hot-entry eviction rules;
preview streams do not replace durable execution history.

`InMemoryCacheBackend(max_bytes=..., max_size=...)` provides hot content and
buffered writes with one shared capacity budget, including in-flight writes.
Hot entries may be evicted to admit work; accepted pending entries stay until
acknowledged. `InMemoryStreamBackend` is a stream-only adapter.
`FilesystemCacheBackend` and `TieredCacheBackend` provide hot content only. A
host can expose any complete combination on its existing client instance.
The framework borrows it and never closes its host connection.

The buffer group uses existing identities and immutable bytes:

- `append(scope, id, key, content, *, wait)` returns whether admission succeeded
- `read_batch(scope, *, limit)` returns ordered `(id, key, content)` tuples without removing them
- `ack(scope, ids)` removes only those accepted IDs

`content=None` is a queued deletion; hot-content absence remains separate.
Adapters preserve same-ID retry identity and exact-ID acknowledgement.
`False` means definitely not admitted. An entry that can never fit returns
`False` immediately even with `wait=True`; only temporary fullness waits.
`StorageBuffer.enqueue` reports this definite rejection as a capacity error
without disabling later valid admissions. Cancellation before admission
publishes nothing; once admitted, the actual admission must settle before
cancellation or close can release ownership. An append error can mean unknown
admission, so it must not be reported as a definite capacity rejection or
silently redirected into a local queue.

## Automatic Metrics buffering

```python
from linktools.ai.observe import Metrics
from linktools.ai.core import CommitPolicy
from linktools.ai.storage import InMemoryCacheBackend

cache = InMemoryCacheBackend(max_bytes=8 * 1024 * 1024, max_size=1024)
metrics = Metrics.sqlite(
    "metrics.sqlite3",
    commit=CommitPolicy.batch(buffer=cache, batch_size=64, flush_interval=15.0),
)
recorder = metrics.recorder()
try:
    accepted = await recorder.try_record(observation)
    result = await recorder.flush()
finally:
    await recorder.close()
```

Automatic Agent and Task observations, including evaluation work in a Runtime,
require `Runtime.open(..., metrics=dataset)`. Runtime owns one isolated recorder;
explicit application recording uses the dataset independently.

All Metrics constructors and factories accept `commit`, defaulting
to `CommitPolicy.batch(batch_size=128, flush_interval=60.0)`. Use
`CommitPolicy.sync()` for immediate writes; Metrics supports sync and batch
policies. A Metrics batch policy requires an integer `batch_size`
from 1 to 256 and a finite positive `flush_interval` in seconds; booleans are
invalid for both.
These settings are validated when creating the dataset and inherited by each
fresh recorder, including those owned by separate Runtimes. Only automatic recorders
use `commit.buffer`; without a buffer they await the source in the caller,
without a hidden local queue. `record`, `record_observations`, and queries keep
their strong source semantics. Queries never add pending observations.

`MetricRecorder.status()` returns `MetricRecorderStatus` in both direct and
buffered modes. `persisted` counts confirmed source writes; `unconfirmed`
counts settled attempts without that confirmation, including cancellation or
an unknown outcome after the source may already have committed. It does not
mean those observations were definitely lost. `MetricFlushResult.completed`
reports prefix settlement, not that every observation persisted.

`BatchWriteScope` is the lower-level scheduling and settlement extension for
custom encoded batch sinks. The injected cache owns pending bytes and capacity;
the scope owns dispatch and its lifetime. Its `BatchWriteStatus.confirmed`
means only that the sink returned successfully. Metrics maps that count to
`persisted` because its sink confirms authoritative writes.
Its `BatchWriteFlushResult` reports settlement of the captured admission prefix,
with `unconfirmed` preserving failed or unknown source outcomes.
`StorageBuffer` separately provides final-value reads and writes over a source;
it reuses the same scheduler rather than defining another pending queue.

Automatic batches flush at the configured entry threshold or interval after
the first pending admission, whichever comes first (defaults: 128 entries and
60 seconds); explicit flush and close wake partial batches. These
are batching defaults, not query TTLs. The cache contains the encoded pending
bytes using the existing Observation codec. Independent recorders own separate
scopes and close only their own work; shared cache capacity remains shared.

Admission is best effort, not a durability receipt. A full cache rejects new
automatic observations; failed/cancelled writes remain visible in status.
`queue_size` includes unacknowledged in-flight entries; `queue_capacity` is
`None` because capacity belongs to the adapter and has no common query contract;
it does not mean unbounded storage. A successful sink is never
called again merely because acknowledgement failed; an unresolved cleanup
failure is exposed by flush/close. Caller cancellation does not abandon
close-owned I/O, and neither recorder close nor Runtime close closes the cache.
`pending` counts source outcomes not yet settled; `queue_size` counts entries
not yet acknowledged in the cache. A successful sink followed by an ack failure
can therefore have `persisted=1`, `pending=0`, and `queue_size=1`; flush/close
report that cleanup failure without claiming the source write failed.

## Asset content digests

`AssetInfo.digest`, `AssetVersionRef.digest`, and `CapabilityLoadEntry.digest`
are the SHA-256 of the asset bytes, encoded as 64 lowercase hexadecimal
characters. Asset version payloads, SQL JSON payloads, filesystem metadata and
history, batch receipts, and snapshot manifests use the same `digest` field.
Runtime bindings preserve it through the Asset version codec. Empty assets and
non-normal entries use the SHA-256 of empty bytes.

`AssetStore` binds its overlay's value validation to `AssetInfo.digest` and
`size`, in addition to any supplied validator. This applies to cached values,
origin reads and write results on that shared overlay. A self-consistent cache
envelope is insufficient when its bytes do not match the selected metadata;
invalid cached bytes are discarded, while an origin mismatch raises an integrity
error before cache publication.

Snapshot content keys bind to the asset digest; each manifest has its own
digest and object reference computed from the complete manifest bytes. HTTP
ETag validators are a separate transport concept.

## Fixed Asset versions

Live `AssetStore` instances backed by an overlay continue to read historical
versions and verify each reference's size and digest through `read_versions`.
Calls admitted before the same source read begins can share that observation. Once historical I/O has started, a later independent
call reads the source again; completed version results are not kept in the byte
cache. Historical existence cannot be checked separately with the current
backend contract, so this reduces same-turn duplicate work rather than promising
warm historical reads without source I/O. Closing the store drains admitted
version reads before closing their backends.

A capability capture retains its existing immutable-byte memo for that capture
only. Concurrent first reads of a captured key coordinate its fill; unrelated
keys can still load independently. This does not share completed values across
captures or expand the capture's lifetime.

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

Overlay reads admitted before close finish before their source backends are
closed. New reads are rejected once close starts, and concurrent close callers
share the same cleanup. Cancelled readers still leave any shared cache or
metadata loads owned by the overlay until those loads settle.

`StorageOverlay.invalidate()` discards materialized metadata after an external
mutation. It is not a global cache flush or a promise of cross-process
read-after-write consistency. `refresh()` follows the configured revision
source and layer policies; it is not a universal forced-origin read.

## Boundaries

Runtime authorization, leases, cancellation, commit confirmation and recovery
remain authoritative operations. Cache hits do not grant access. Ordinary
observations may retain deleted or expired source members only within their
explicit bounded observation window; strong consumers keep current source
validation. Transient or volatile content must not gain persistent filesystem
or remote copies merely because its bytes are immutable.

Live event delivery and reconnect cursors are separate from content caching.
A shared byte cache does not provide event ordering, subscription recovery, or
producer fencing. Likewise, business TTL snapshots and explicit refresh
semantics remain with their read owner unless an equivalent contract has been
established. Existing Runtime history calls do not automatically acquire shared
content caching from the asset-store configuration above.

## Runtime state observations

Built-in `StateStore.read_record(key)` and `read_records(keys)` use the same
optional hot-cache group. Records and a complete batch share an absolute
three-second capture deadline; hits require no source query. The namespace,
tenant and logical store identify the source view. Sharing the cache therefore
requires the source-binding rule below, including for filesystem and SQL stores.
No cache means direct source reads.

`read(callback)` and `mutate(callback)` retain their source transaction semantics;
record reads inside those scopes bypass the cache. Successful local mutations
invalidate that owner's observations without letting an older in-flight load
repopulate them. Other owners may observe the old snapshot within three seconds.
A custom `StateStore` can implement the two ordinary methods by delegating to
its existing `read` callback; caching is optional, correctness is not.

After a confirmed SQL mutation, complete `session` and `execution` records
written or guarded by that transaction may fill their singleton observation keys.
A no-op mutation neither invalidates nor republishes these observations.
Each store owns at most one publication task; a busy publisher drops additional
offers rather than queuing them. Publication failure cannot change the committed
result, and close joins the owned task's cancellation cleanup. Failed or unknown
commits do not publish. Complete batch reads keep their own source snapshot and
are never assembled from singleton entries.

These fills keep the original transaction capture deadline. Neither commit nor
delayed publication starts another three-second window. Local invalidation
suppresses older local fills; independent publishers may replace a newer shared
entry with an older, still-valid capture. The bounded observation contract does
not promise globally monotonic reads. Fills can avoid source reads at the cost of
extra cache writes and bytes; fewer SQL queries do not imply fewer total network
operations. Other backends keep their existing read-through behavior.

Runtime `inspect` and ordinary display observations can use this bounded view.
Explicit results, waits, next-turn admission, terminal/unknown-commit readback,
and effect ownership use their committed owner paths. These guarantees do not
require a new per-call cache option or a different Runtime profile.

## Runtime observation publication

Metrics uses `commit.buffer` with its configurable scheduler (defaults: 60
seconds or 128 entries). Trace uses the same owner-scoped buffering primitive
through its own `commit.buffer` (defaults: three seconds or 128 observation
boundaries). Without a buffer, each observation boundary awaits the existing
guarded projection commit. The route's hot cache never enables write batching.

The policy borrows its buffer; equality and hashing compare timing values only.
Within one Runtime, owners using the same buffer object and equal batch size
and interval share automatic flush scheduling. Their accepted, undispatched
entries contribute to one threshold; the earliest pending entry sets the
next deadline, which later admission cannot postpone. A trigger dispatches each
owner's currently accepted prefix. Entries count queue records: a Trace marker
is one observation-boundary flush request, not one SQL row.

Each owner keeps its queue, callback, acknowledgement and domain failure
semantics, including unknown-result readback. Source transactions remain
separate, even when storage points to the same database. Explicit flush and
close affect only that owner; they do not wait for unrelated Metrics delivery.
The last departing owner cancels the shared timer. Different Runtime instances,
buffers or timing values retain separate schedules. Standalone recorders remain
independent. Sharing a buffer does not imply cross-process claiming.

A Trace queue entry is a flush request, not the sole copy of trace evidence.
The run's existing staging owner retains facts until atomic publication and
readback confirm them. Queue acknowledgement is not a durability receipt and
this owner-local queue does not provide cross-process recovery of pending trace.
Capacity rejection or cache failure settles owned callbacks before direct
publication; a source integrity error or unresolved commit remains authoritative.

Required flushes, checkpoints, effect fences and terminal publication retain
their existing boundaries. Before terminal capture or revoked-producer discard,
queued requests are quiesced without independently committing the dirty tail.
Already-running publication settles first. Closing a run drains its owned work
without closing the borrowed cache.

## Runtime content

Runtime storage accepts a borrowed `cache` independently of asset
storage. The ordinary history and session APIs continue to own authorization,
retention and reference selection; their external object reads use the storage
router. Inline payloads do not enter the object cache. Runtime already owns its
ObjectRef/domain/tenant/source rules, so callers supply the byte backend directly;
there is no separate Runtime content adapter to construct.

```python
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.storage import InMemoryCacheBackend

cache = InMemoryCacheBackend(max_bytes=16 * 1024 * 1024)
storage = RuntimeStorage.from_root("./runtime", cache=cache)
```

The host can retain the cache across successive storage lifetimes. Different
storage instances share cached bytes, not their in-flight loads. The owning
storage settles its own reads on close without closing the borrowed cache or a
host connection pool. Closing first completes pending archive projections,
then drains accepted object reads before source storage is closed.

Only durable routes use this cache. A cold or expired read uses the source
and verifies the expected size and digest. A hot read verifies those immutable
bytes again without an object-metadata query. Source membership may be stale
for at most three seconds from the original load's start; expiry checks live in
the owner envelope, so promoting bytes into an L1 without TTL cannot renew them.
Deletion or source damage is detected on the next source read. A caller arriving
after a shared load's deadline starts a new source read; a slow original read
may finish for its original callers without restarting indefinitely.

Recovery-domain objects always retain current source-membership checks. Other
strong consumers retain them as well: explicit result/wait,
RuntimeHistory results, persistent checkpoint reads, model/recovery context loading, and every rebuilt session
display (including `max_age=0`). They may reuse verified bytes after the existing
ObjectStore `stat` validation instead of transferring the complete object again.
This does not claim to detect physical corruption beyond that ObjectStore
contract. Authorization and control checks remain with their domain owners.

Sharing a cache requires each Runtime namespace to identify the same logical
data source. When changing databases or object sources, use another Runtime
namespace or a separate cache namespace/instance. Store IDs alone do not
identify physical sources. Keys also isolate tenant, domain, and the complete
immutable ObjectRef. A host-provided cache is a trusted byte transport, not an
authorization service.
`RuntimeState.read_object(domain, reference)` is the composition-level read
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

## Ordered live streams

`ExecutionCapability(coordination=cache)` uses the optional `open_stream`/`read_stream`
methods on that same borrowed instance for ordered, disposable live previews.
`object_store` still owns authoritative payloads; the separately configured
StorageRoute cache holds reproducible content. A live preview
is not a successful durable commit. Configure shared authoritative execution
storage as well as a shared stream backend to observe another Runtime's live
output. Execution routes must have `DURABLE` retention; temporary execution
storage cannot supply the authoritative history and recovery boundary.

Local producers keep their in-process subscriptions even when a shared backend is
configured; a shared preview outage does not remove their live text. Shared
publication remains available to remote observers. A remote preview cursor moved
to a local producer emits `STREAM_RESET` before replaying from its durable
position, because transport positions cannot address the local preview buffer.
Standalone execution watches are owned by their Runtime: close cancels and drains
pending reads and paused iterators before releasing storage. Explicit iterator
close remains supported and does not cancel the execution producer.

Without a complete stream method group, Runtime keeps its existing local streaming path.
The SDK does not construct Redis clients or require Redis. An adapter implements
`StreamBackend.open_stream/read_stream` and the returned `StreamWriter.append/seal` operations.
Runtime supplies keys, producer identities, sequence numbers and encoded frames;
application run/watch/cancel callers do not construct them.

`InMemoryStreamBackend(max_bytes=...)` is an event-loop-local reference adapter.
Its shared logical payload budget covers all streams, with at least one byte
charged per record and empty stream. It is not an RSS or physical-disk limit.
Writes evict other least-recently-written streams before trimming the current
stream. Missing or trimmed records produce an explicit gap. `StreamBatch.sealed`
is true only when the returned batch reaches the sealed tail, so a limited read
must consume preceding batches before stopping.

Adapters must preserve sequence order and same-payload retry idempotency for retained
records (retries of trimmed records fail closed), reject
conflicting retries, and atomically append the seal record while refusing future
writes. An evicted stream invalidates its previous writer. Runtime never retries
an unknown open or recreates a lost stream under the same admission; observation
falls back to authoritative history. A later admission has a distinct generation.

The borrowed adapter's connection remains host-owned. Runtime drains its own
read and publication work before releasing its resources; closing one Runtime
does not close the shared client or delete another Runtime's streams. Host
adapters must bound retained data and make blocking reads cancellable.

Remote text deltas are provisional; semantic events are shared after their
existing durable confirmation boundary. Reconnection must reconcile previews
with canonical history rather than append the same final content twice.
Generation and terminal checks run independently of incoming text, backing off
from one second to a maximum interval of 30 seconds. Lost notifications can
therefore take up to that interval plus authoritative I/O to converge. A shared
log does not remove child-discovery scans or guarantee lower SQL counts.

This transport does not route control requests to an execution's owning
process. Existing unsuccessful/unknown non-owner cancellation results remain
meaningful. Cancellation uses execution ownership and effect-ledger rules,
including explicit force revocation where supported; stream delivery does not
stop a remote provider or prove cancellation or task migration.

After a preview gap or admission change, the current watch uses durable replay
for the remainder of that observation. A cursor containing only durable
progress does not silently resume provisional text from another admission.
An initial watch opened before any producer claim instead follows durable
events until the first attempt is admitted, then may attach to that claim's
shared preview. It does not attach to a later attempt or a changed claim.
Realtime continuation across every recovery is therefore not guaranteed; the
reset and authoritative history are the safe recovery result.

Preview cursor continuation applies to execution watch and its model observation
projection. TaskGraph observation retains its durable event replay contract;
configuring the backend does not add provisional TaskGraph cursor semantics.

## Opt-in session display captures

`runtime.sessions.refresh(session_id, principal=principal)` remains a strong
refresh. An explicit `max_age=3` permits reuse of one complete `SessionDetail`
for at most three seconds from the start of its original capture. `max_age` must
be finite and non-negative; it is a caller-selected maximum, not a fixed SDK TTL.
A shorter later maximum is respected, and a longer one never extends an existing
entry's original expiry. The default zero does not read or publish display cache
entries. Custom `SessionService` implementations accept the same `max_age` argument.

The Runtime borrows the effective Conversation route’s cache; no extra cache
configuration or backend lifecycle is introduced. Standard assembly enables
this optimization only when conversation and execution retention are both
`DURABLE`. Entries are isolated by Runtime namespace, principal/tenant/kind,
session, cursor, and page limit. Instances sharing that logical source can reuse
the same capture. Its wall-clock capture and expiry travel with the payload;
reopening a Runtime or reading another cache tier never renews its deadline.
Hosts sharing cache entries must keep clocks synchronized. Different sources
require separate namespaces or cache instances.

Every hit still reads the authorization header, runs the current policy outside
the metadata transaction, and freshly checks the session, the referenced
executions and their binding identities, and conversation history/head. An
identity/ownership mismatch or missing resource cannot be revived by the cached page. Forks,
ancestor cursors, custom repositories, and split metadata routes keep the
original complete read path; this preserves ancestor deletion/ownership checks.
Reads inside an existing state transaction bypass the display cache entirely.

A hit returns the entire original display, without mixing fresh status or
metadata with an older timeline/body. A changed session revision or active
execution binding, or a cached active execution that has become terminal,
invalidates the capture and refreshes the complete display. These checks reuse
the existing fresh reads; a session revision is not a version of every display
dependency. Other execution display changes and new turns that do not change
those fields may remain undisplayed until expiry or a strong refresh.
Independently removed/corrupted transcript facts or object bodies may likewise
be detected up to `max_age` later. Sending, cancelling,
resuming, and all other control operations retain their authoritative reads.
The current authorization policy runs on every call; the dependency validation
reads source records directly, so display expiry is not stacked on top of a
metadata-cache expiry. Rebuilding a page also validates current object membership,
so an older body-cache capture cannot acquire a new page lifetime. Signed input and output cursors are checked again and
are not renewed by a cache hit.

Cache serialization uses the existing domain leaf codec without adding durable
record types. A private payload checksum detects cache byte corruption (it is
not an authentication signature), and typed views are reconstructed per reader.
Cache errors or malformed entries fall back to the source; source errors and
cancellation propagate. All cache I/O is awaited outside metadata transactions,
with no background refill, detached task, or borrowed-backend close. Runtime close
rejects new calls and drains admitted display-cache reads before closing their
sources. Cancelling a close waiter does not cancel that owned drain; a later
close can await it. A custom backend that does not settle can delay close, rather
than being reported closed while cache I/O is still in progress.

## Readable final-value buffering

`StorageBuffer` adds an explicitly buffered view over an initialized byte-valued
store with `get` and `apply_batch`. Existing `AssetStore` and `StorageOverlay`
can supply these operations; a custom store implements the same structural
contract. The chosen store must support its batch operation. In particular,
`AssetStore.apply_batch` requires an atomic-capable Asset backend; the buffer
cannot create a capability missing from the source.

```python
from linktools.ai.storage import InMemoryCacheBackend, StorageBuffer

buffer = StorageBuffer(
    assets,
    cache=InMemoryCacheBackend(max_bytes=8 * 1024 * 1024, max_size=1024),
    read_pending=True,
)
await buffer.enqueue(key, b"first")
await buffer.enqueue(key, b"latest")
preview = await buffer.get(key)
await buffer.flush()
await buffer.close()
```

Without a buffer-capable `cache`, `enqueue` directly awaits the existing source
write; no private queue is created. With buffering enabled, `enqueue` waits for
admission capacity and returns only after acceptance. It
does not return a durable receipt. `enqueue_delete` accepts a deletion in the
same order. Closing or a failed buffer rejects new writes explicitly rather
than returning an easily ignored boolean. Cancellation while waiting for
capacity does not accept the waiting mutation.

`read_pending` is selected once at construction. The default `False` exposes only
source-confirmed values, with a three-second read-through hot view when the cache
provides the complete hot group. This means confirmed data, not the latest source
state on every call: another owner or external writer can remain invisible until
three seconds after the original capture. Hits and tier promotion never renew
that deadline. `True` first overlays the latest accepted,
unresolved local value or deletion, then uses that committed view. It does not
expose another buffer's pending writes or call an unconfirmed value committed.
After a successful batch the local hot entry is invalidated; the next read loads
and caches the confirmed source once. This optional refill cannot change the
write receipt or delay acknowledgement. In pending-visible mode, a source read overlapping a
complete enqueue/commit/settlement cycle is rechecked rather than returning its
obsolete result. Committed-only mode preserves one source read's snapshot
contract; it does not retry to chase later concurrent commits.
Domain authorization still belongs to the domain owner before using this
storage capability.

This is a final-value buffer: repeated keys within one dispatch can collapse to
the last value. Intermediate values need not be persisted. The source receives
ordinary `StorageChange` PUT/DELETE entries; conditional revisions, RESET,
metadata changes and business atomic batches are not implicitly queued.
`Metrics` continues to use append buffering, where every observation matters.

A successful `flush` confirms that intentions accepted before the call have
been persisted, or superseded by a later same-key value that was successfully
persisted. It does not promise an intermediate version or a transactional
snapshot. A failed or unknown write stops further acceptance and is reported
by `flush` and `close`; it is not silently retried or described as rolled back.
Reads then use actual source data, which may include a write whose response was
lost. Queue entries never dispatched are not called committed.

All writes for a buffered key must follow the same owner ordering. Writing
straight to the source while older buffered work remains can allow that older
work to overwrite the new value. Source `put` retains its existing durable
contract, but it is not an ordering bypass for this buffer.

The buffer borrows its store and does not close it. Close stops admission and
drains accepted reads and real sink work. Cancelling a close waiter does not
abandon owned work; wait for close to finish before closing the borrowed store.
An uncooperative sink cannot be reported quiescent merely because a timeout or
cancellation was requested. The injected cache owns pending bytes and its capacity includes in-flight work;
pending views reference those immutable values rather than introducing a second
payload queue. A logical byte budget does not promise an RSS or physical-disk limit.

A valid committed hot hit does not query the source. On a miss, committed reads
retain the store's existing `ContentCache`, metadata and source-validation
behavior. The hot view is scoped to this buffer owner; it does not promise
cross-owner immediate visibility or act as an authorization cache. Native SQL
batching, transaction-local record reuse, CAS and atomic Runtime commands stay
with the authoritative store. No execution control state is moved into this
best-effort buffer merely because it shares a database.
