# Storage caching

Caching accelerates reads; the source storage still owns durable facts. The
storage package separates content, revision discovery, and metadata projection.

`PersistenceStorage` declares reusable resource routes and an optional object store.
Each Persistence binding acquires and closes its operational scope. Runtime and
TaskEngine borrow it and own only their producer-local lifetimes; borrowed
backends remain host-owned. Stores
such as `AssetStore`, `ObjectStore`, and `MetricStore` expose their respective
data contracts; repositories own domain records and state transitions. Backends
provide replaceable source or transport capabilities. These roles are not
determined by whether a type is a protocol or a concrete implementation, and a
store does not necessarily imply durable retention.

`Runtime.open` and `Runtime.attach` register their lifetime with Persistence
before composing producers. An execution filesystem route admits only one Agent
Runtime on that Persistence; Task-only Runtimes do not occupy that exclusive
execution slot. SQL and Memory continue to allow independent Agent Runtimes.
`TaskEngine(...)` and `TaskEngine.open(...)` also register a borrowed lifetime.
Definition views share that registration; closing a view does not release the
engine. Task producers do not occupy the filesystem Agent-exclusive slot, and
read-only Task observation remains available.

`Persistence.close()` rejects active Runtime or TaskEngine borrowers without
closing their resources. An owning Runtime closes Persistence only after its
own cleanup and after independent producers have released it. TaskEngine
releases only after resource cleanup and every close callback succeeds. Failed
cleanup or notification retains the registration for retry; cancellation of
the caller does not release an in-flight cleanup.
This is instance-local protection, not a cross-process lock or a new SQL
takeover mode. Callers must still finish their own in-flight ordinary Task
queries before closing the engine. History is not registered by this producer
mechanism; hosts must await its ordinary queries and close its subscriptions
before closing Persistence. Filesystem state retirement and retained-history migration
require their separate recovery and inventory checks.

`PersistenceStorage.from_root(root)` selects `root/runtime.db` and file-backed
ObjectStore bytes; it does not migrate filesystem state. When that database is
absent, the factory refuses a root containing recognizable filesystem state
manifests instead of silently creating an unrelated empty database. Quiesce the
old writers and explicitly export/restore supported state into a separate empty
root, then verify the retained records and object references before switching.
Portable snapshots preserve the core Memory domain, including its operation
receipts and object-backed content. Arbitrary durable capability partitions remain
unsupported; this guard does not add their migration support.
An existing `runtime.db` remains the selected SQLite source even if old files
coexist. This check neither verifies that database as a completed migration nor
recognizes every damaged/custom layout, and is not a cross-process ownership
barrier against a root being changed after configuration.

| Interface | Responsibility | Host integration |
| --- | --- | --- |
| `CacheBackend` | Store discardable immutable bytes | Supply it to a domain content cache or `PersistenceStorage.sqlite(path, cache=...)` |
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

`StorageRoute(..., cache=...)` and the PersistenceStorage factories borrow a
`CacheBackend` for hot reads only. `CommitPolicy.batch(buffer=..., ...)` and
`CommitPolicy.terminal(buffer=...)` borrow an independent `WriteBufferBackend`
for pending payloads. The same implementation
may provide both capabilities, but each use is explicitly configured.
`PersistenceStorage(..., coordination=...)` separately supplies turn ownership and
live streams, including the non-authoritative pending-turn view.

### Owner defaults for batch writes

Provide only the batch settings you want to override. A buffer-only declaration
uses Trace's 128-entry / 3-second defaults:

```python
from linktools.ai.capability import TraceCapability
from linktools.ai.core import CommitPolicy

trace = TraceCapability(commit=CommitPolicy.batch(buffer=write_buffer))
```

Here `write_buffer` is a host-provided `WriteBufferBackend`. In **batch mode**,
omit `buffer` for synchronous source writes without a hidden queue.
`terminal()` without a buffer instead retains the owner's existing local staging
until its publication boundary. Explicit `batch_size` and `flush_interval` always win;
for example, `CommitPolicy.batch(buffer=write_buffer, batch_size=32)` changes
only the size and retains the receiving owner's interval.

`None` for either timing field means an omitted authoring value. Trace resolves
those fields once in its constructor and retains a complete,
immutable policy before any Runtime capture, recorder or write scope uses it.
The input policy is not mutated, so it can be reused by owners with different
defaults. The generic policy has no universal batch interval.

A custom `CapabilityPersistence` owner chooses its own defaults in its
constructor with the same public operation:

```python
self._commit = commit.resolve_batch(batch_size=32, flush_interval=5.0)
```

Expose that complete policy through the capability's `commit` property before
registering it. Runtime rejects custom batch policies that still omit either
timing field rather than guessing defaults. `resolve_batch` leaves sync and
terminal policies unchanged and preserves explicit batch settings and the
borrowed buffer. It does not change safety writes, transaction grouping, or
flush/close ownership.

### Runtime archive routes and defaults

`PersistencePlan.checkpoint=None` and `trace=None` inherit the execution route,
including its hot-read cache. Registering CheckpointCapability or TraceCapability
enables behavior; neither capability accepts a physical storage argument. An explicit archive route must remain
in the same `StateStorageGroup` as Execution and Recovery, because their safety
and terminal boundaries require atomic writes. Persistence validates this when it opens. Routes share a group when they use the same SQLite path, the same SQL
engine instance, or the same filesystem transaction root; memory routes share
one binding-local group. Separate SQL engine instances targeting the same database do not
establish a shared SQL group.

For a per-archive hot cache, derive the execution route rather than selecting
an unrelated database. This current API example keeps Checkpoint's inherited
cache while disabling Trace's hot cache:

```python
from linktools.ai.capability import CheckpointCapability, TraceCapability
from linktools.ai.runtime import RuntimeFeatures
from dataclasses import replace
from linktools.ai.runtime import Runtime
from linktools.ai.core import PersistenceDomain
from linktools.ai.persistence import PersistenceStorage

storage = PersistenceStorage.from_root("state", cache=hot_cache)
execution_route = storage.plan.route(PersistenceDomain.EXECUTION)
features = RuntimeFeatures("app")
features.capability(CheckpointCapability())
features.capability(TraceCapability())
storage = PersistenceStorage(replace(storage.plan, trace=execution_route.with_cache(None)))
async with Runtime.open("app", storage=storage,
    features=features,
) as runtime:
    ...
```

Here `hot_cache` is a host-provided `CacheBackend`; model bindings, if needed,
are registered with `features.model(name, binding_or_loader)`. `with_cache(None)` changes only the hot-read resource, preserving
the route's transaction identity. It does not disable Trace publication or
Checkpoint safety writes. To choose a different Trace hot cache, pass that
backend to `with_cache` instead.

Missing write buffers make batch policies await source publication without an
implicit queue. `ExecutionCapability` is optional: when omitted, Runtime uses
sync commit; any coordination resource comes from PersistenceStorage. A terminal preference must
be declared explicitly and remains subject to coordination and safety rules.
Runtime drains its own work on close, but never closes borrowed hot caches,
write buffers, coordination resources, SQL engines, or Metrics datasets.

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

## Automatic Metrics providers

`Metrics` is a synchronous source/query dataset and a default
`MetricRecorderProvider`. Automatic recording can instead use a host provider
without creating a SQL dataset in the producer:

```python
from linktools.ai.observe import MetricRecorderProvider, Metrics

metrics = Metrics.sqlite("metrics.sqlite3")
recorder = metrics.recorder()
try:
    accepted = await recorder.try_record(observation)
    result = await recorder.flush()
finally:
    await recorder.close()

# telemetry implements MetricRecorderProvider over an already-open host client.
async with Runtime.open("app", storage=storage, features=features,
                        metrics=telemetry) as runtime:
    tasks = await runtime.tasks.bind(*definitions)
```

The provider contract is `recorder() -> MetricRecorder`, with no arguments.
Every call returns a fresh scope, owned by its Agent or Task producer.
Construction starts no I/O or background tasks and acquires no asynchronous
resources. A host recorder may start work on its first `try_record`; its close
must settle its own work without closing the provider/client or another scope.
Evaluation borrows its TaskEngine and does not create another recorder.
`metrics=None` disables automatic observations without creating any dataset.

`Metrics` constructors/factories have no commit policy, and `recorder()` has no
batch-scheduler parameter. Both explicit writes and its default automatic
recorder await source settlement; no SDK queue, timer or threshold is created.
Queries never include pending host observations. Runtime continues to enrich
observations with dimensions and correlation while preserving their original
IDs, source namespace and tenant. The provider binds the target dataset; those
business-source fields must not silently choose a different destination.

A host may implement bounded publishing and a separately managed consumer.
Only the host chooses batching, capacity, retry and client ownership. Broker
acceptance is not a SQL receipt: `persisted` counts confirmed source writes,
`unconfirmed` may include successful broker delivery awaiting source proof,
and `flush.completed` means the recorder's accepted prefix settled, not that
all consumers committed. Publishing success without SQL proof is not itself a
write failure. Unknown outcomes must remain visible; do not silently fall back
to per-observation SQL or disable recording when the broker fails.

The existing `MetricRecorder` methods (`try_record`, `status`, `flush`, `close`)
remain unchanged. A direct recorder completes admission after its source call
settles; flush does not wait for calls still in admission, while close stops
new admissions and waits for active calls without cancelling their callers.
Failures are best effort and cancellation propagates without rewriting already
confirmed business facts. SDK owners close scopes, never the borrowed provider.

### Migrating automatic recording

Remove `commit=` from Metrics factories and `batch_scopes=` from recorder calls.
Default, explicit-sync and buffer-free batch calls were already synchronous;
they retain that behavior. A genuinely buffered caller must switch to a prepared
host provider/consumer in the same coordinated release, not silently become
per-item SQL. Drain the old scopes and preserve failed/unknown outcomes before
switching; owner-local pending bytes are not a recoverable work queue.

Host consumers retain stable observation IDs and submit through
`Metrics.record_observations`. Advance acknowledgements only after source
confirmation; duplicates, unknown commits, conflicting payloads, retention and
consumer reassignment need their own tested settlement policy. Producer-only
provider support does not prove any broker adapter or deployment is ready.

`BatchWriteScope` and `StorageBuffer` remain available to their other owners.
Their generic settlement status is not a reliable work-queue acknowledgement
contract. Trace continues to use its own commit policy and lifecycle, independent
of automatic Metrics.

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

Write timing and storage are independent. `sync()` confirms source writes at the
owner's existing boundaries; `batch(buffer=...)` schedules admitted payloads;
`terminal(buffer=...)` defers optional payload publication to the terminal
boundary. Terminal has no batch size or timer. `terminal()` without a buffer
keeps local staging; batch without a buffer awaits source writes. Neither a
hot-read cache nor a coordination backend implicitly enables payload buffering.

### Payload ownership and publication

Execution, Trace and explicit `CapabilityPersistence` writers keep independent
scopes even when they borrow the same backend. Entries carry immutable serialized
business data, not `None` flush markers or digests that require an unacknowledged
RAM payload mirror. Execution captures its deferred records, input/context,
instructions, output and audit facts, including the retained turn context when
required. Trace captures logical projections and their selected request/response,
failed-stream and object-content dependencies. Already durable immutable object
references keep their existing ownership. New object bodies are materialized
when the owning source publication is prepared, not merely to encode a buffer
entry. Necessary durable admission/checkpoint/effect preparation may occur earlier.

Publication reads and verifies the complete fixed admitted prefix from the
backend before source preparation. Missing, short or corrupt reads do not permit
using newer staging, publishing a smaller prefix or acknowledging it early.
Later cumulative snapshots cannot replace previously admitted append-only facts.
Source sequence assignment, CAS and producer fencing remain in the authoritative
transaction. Compatible terminal prefixes join the existing Execution terminal
transaction; each distinct participant scope contributes its full selected prefix.
There is no distributed transaction across unrelated storage resources.

Within one Runtime, batch Trace runs using the same buffer object and equal batch
size and interval share automatic scheduling (defaults: 128 observation
boundaries / three seconds). Accepted undispatched entries count against that
threshold; the earliest pending entry sets the deadline. Each run still owns its
transaction and failure handling. Other Runtime instances have independent
schedules and scopes. Policy equality excludes buffer identity; grouping writers
must compare the borrowed instance separately.

### Admission, failures and cleanup

Admission is awaited with `wait=False` and bounded transport waits; terminal
producers must not wait for a terminal drain that cannot run until they finish.
Capacity rejection or unresolved append failure fails that write/turn rather than
silently falling back to ordinary SQL publication, disabling observations, or
repeating the model request. An uncertain append retains its original ID and
bytes; readback must resolve that exact entry, not create a replacement ID.

For Trace, terminal-buffered Execution and buffered capability owners, a known rejection
can isolate the rejected tail only after verifying the complete accepted prefix.
Already-admitted Execution/Task control must still
settle safely as FAILED or CANCELLED; it must not remain STARTED merely because
optional bytes were rejected. Batch Trace can confirm its accepted prefix before
that control settlement; terminal Trace joins its accepted prefix to the terminal
transaction. Unadmitted output is not copied into SQL as a fallback. A turn with
no durable admission may fail and release its pending coordination instead.
Unknown admission or source outcomes are not treated as known rejection.

Known source rollback retains the fixed payload for explicit retry. Unknown
source outcomes retain the original plan, IDs and source flight until canonical
readback resolves them. Only source-covered IDs are acknowledged. After source
confirmation, acknowledgement is cleanup: failure does not change canonical
success/failure/cancellation, replay source writes, or prevent the control result
from becoming visible. The owner retains cleanup for retry. Close reports
unsettled source/control or cleanup failures and never closes a borrowed backend.

An ordinary explicit Trace publication boundary admits the current dirty tail
before draining it; it must not claim success for a rejected tail. Once a terminal
plan owns a fixed prefix, flush/close cannot downgrade it to an unrelated batch
write after a failed terminal attempt. Canonical settlement or explicit authorized
revocation must resolve that ownership. Revoked-producer discard first quiesces
capture/publication and requires the existing source revocation proof. Required
Task, recovery-checkpoint and tool-effect writes keep their own durable boundaries.

### Reading and deployment limits

Working reads remain with the existing owner-local staging/overlay. Independent
History readers see source-confirmed data, not arbitrary buffer entries. The
buffer does not replace SQL retention, hot-read cache policy, authorization,
coordination leases or effect receipts. It does not provide process-restart
handover, another Pod's Python callbacks, or Redis-only recovery. Capturing bytes
does not by itself unload RAM mirrors or promise zero-copy execution.

Automatic Metrics remain independent best-effort observations through their
provider; they do not own Trace capture or swallow a Trace admission error.
`metrics=None` does not disable the Trace interaction recorder. Metrics source
queries remain committed-only. Synchronous SQL publication plus bounded hot-read
caching remains a valid separate configuration; cache hits do not authorize
control transitions or resolve unknown source commits.

**Adapter/capacity migration:** stop and drain old marker producers before
upgrading; their active scopes are not reusable with the payload format. Size
capacity for real serialized bytes, cumulative snapshots and content context,
not boundary counts. A host backend must retain accepted entries and provide the
requested ordered prefix non-destructively until exact-ID acknowledgement. Shared
capacity pressure must not evict unconfirmed payloads as ordinary hot-cache data.

**Verification scope:** local directed tests exercise payload readback, rejection,
rollback/unknown outcomes, acknowledgement failure, cancellation and independent
Runtime lifetimes using the in-memory buffer and local source backends. These do
not certify a Redis adapter, host failure/reassignment behavior, memory savings,
or production throughput. Complete Execution/Trace/Capability combinations and
supported source/dependency matrices require their own exact-head integration
and CI evidence; the policy declaration alone is not that evidence.

## Runtime content

Runtime storage accepts a borrowed `cache` independently of asset
storage. The ordinary history and session APIs continue to own authorization,
retention and reference selection; their external object reads use the storage
router. Inline payloads do not enter the object cache. Runtime already owns its
ObjectRef/domain/tenant/source rules, so callers supply the byte backend directly;
there is no separate Runtime content adapter to construct.

```python
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.storage import InMemoryCacheBackend

cache = InMemoryCacheBackend(max_bytes=16 * 1024 * 1024)
storage = PersistenceStorage.from_root("./runtime", cache=cache)
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
History results, persistent checkpoint reads, model/recovery context loading, and every rebuilt session
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
`Persistence.read_object(domain, reference)` is the composition-level read
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

## Source-backed turn authority

`linktools.ai.persistence.turns.StateTurnCoordination` implements turn ownership,
request lookup and authoritative session lookup over an initialized `StateStore`.
Use the owning Persistence's public `execution.executions.state_store`; its
session/request/execution records and mutation receipts are included in the
Execution snapshot contract. This is an alternative authority, not a Redis
projection or a second writer alongside an existing coordination backend.

The backend supplies **only** the turn interfaces. Runtime selection still
requires a host-owned object implementing both `CacheBackend` and
`TurnCoordinationBackend`. Compose those interfaces with explicit delegation:
create the host object, pass it to `PersistenceStorage(..., coordination=...)`,
open that single Persistence, bind its ready source backend exactly once, then
use `Runtime.attach(persistence=...)`. Unbound calls must fail, not fall back to
another authority. Carry the Persistence's read-only and ready/closed guards;
never rebind an existing host object to another source or namespace. Runtime
closes before its Persistence; the host retains ownership of the borrowed cache.
No new SDK coordination factory or Runtime mode is required.

Claims atomically maintain session exclusion and request/execution identities.
Lease durations round upward to milliseconds; fences advance per session and
must be compared with the complete opaque lease key and token. Replays and
reads never grant a lease or renew it. Cancellation requires a fresh live
source guard. Expired cross-session request reuse retires all referenced old
generations in the same transaction. Unknown commits are confirmed only by the
original retained receipt; missing receipts are not proof of rollback.

Source observations use independent read scopes and a backend-clock sample.
A consistent empty anchor can report absence at that actual source read, not a
promise that later claims cannot occur. Reachable broken edges remain errors.
This does not detect arbitrary external deletion or a partial resource restore.
Session anchors and execution receipts are retained: budget for their source
writes and storage growth rather than assuming cache-like expiry or zero SQL.

Switching authorities or restoring their records requires stopping old
admission, draining producers, resolving unknown mutations, and moving a complete
source generation. Do not run old and restored authorities concurrently or roll
a source back beneath live effects. A snapshot does not prove those effects
have stopped. Tests cover Memory, Filesystem and SQLite; deployment-specific
strong-source routing, latency and resource capacity still require validation.

## Ordered live streams

`PersistenceStorage(..., coordination=cache)` uses the optional `open_stream`/`read_stream`
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
An observation publisher must preserve every observation rather than use this
last-value coalescing contract.

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

### Optional grouped record reads

Custom state backends do not need to implement SQL query optimizations. The
required `StateStorageGroup` and `StateGroupTransaction` contracts are unchanged.
`GroupRecordReader`, exported by `linktools.ai.runtime.state`, is an optional
operation on a supporting storage group and its active callback:

```python
async def get_records(
    requests: Mapping[StateStore, Sequence[bytes]],
) -> Mapping[StateStore, Mapping[bytes, StoredRecord]]:
    ...
```

Keys belong to explicitly enlisted stores. Call the operation only inside that
group's existing `read_many` or `mutate` callback, from the owning task. Calling
it on the group does not open a new transaction. Unsupported backends keep the
original repository read path; they are not required to emulate an extra
prefetch. Empty per-store results mean observed absence, not unsupported access.

The SQL implementation combines bounded key lookups across partitions and
updates the existing transaction-local record cache, including missing keys.
Subsequent inserts, replacements, guards and deletions update that same view.
The operation neither caches across transactions nor replaces domain CAS,
unique insertion, content integrity or publication checks. Large requests split
at the backend's parameter budget; fewer SELECT statements are not a guarantee
of one network round trip or a transaction-wide snapshot on READ COMMITTED.

Archive preparation may reuse a completed metadata observation for its exact
requested content keys. Existing object bytes are still verified outside the
metadata callback, and publication checks source records again inside the owner
mutation. This observation is separate from prepared writes and verified-body
receipts and is discarded after consumption or preparation failure.


## Conditional record leases

`StateTransaction.transition_record_lease(key, expected_storage_revision=...,
transition=RecordLeaseTransition(...))` changes the existing record's lease
without changing its data or physical identity. `RecordLeaseTransition` is
exported by `linktools.ai.persistence.state`:

- `RecordLeaseTransition("acquire", owner, fence, lease_seconds)` requires no live
  lease and advances the expected fence once.
- `RecordLeaseTransition("renew", owner, fence, lease_seconds)` requires the same
  live owner and fence, and preserves the fence.
- `RecordLeaseTransition("release", owner, fence)` requires the same live owner
  and fence, clears owner/expiry, and preserves the fence.

The backend computes expiry from its current time and returns the actual
`StoredRecord`, including expiry and physical revision. Missing records, stale
revisions and unmet conditions return `None`. The operation does not create a
record; an initial empty lease record can be inserted in the same mutation.
Physical revision increments belong to the backend, not to a domain revision.

SQL first checks the candidate, obtains the existing optimistic revision CAS
write protection, then samples current database time in a separate statement
and checks again. This avoids accepting an expired owner after waiting for the
CAS. PostgreSQL uses `clock_timestamp()` for this operation; `tx.now()` retains
its transaction-fixed semantics. A condition that stops holding during the CAS
wait aborts the mutation with `STORAGE_CONFLICT`, including when the callback
catches it. Existing transaction retry, cancellation and unknown-commit behavior
remain in effect; an unknown commit is not a confirmed acquisition or failure.
The primitive introduces no schema change or explicit lock read.

The condition decision occurs after write protection is obtained, not at commit
or response delivery. A returned lease may already be expired when received.
Ordinary record reads remain snapshots and cannot grant authority to execute;
competing ownership changes must use the conditional operation, and external
effects still require their own fence/idempotency enforcement. Keep mutation
callbacks free of external effects and short. Task and Tool renewal use this
operation; other domain transitions retain their existing authorization checks.

## Offline filesystem member inspection

A filesystem state group's `offline_exclusivity()` retains its existing
`None`-yielding guard contract. On an uninitialized group it acquires the group
(or standalone root) writer lock before discovering members, then acquires all
member locks and validates the complete manifest set. A pending recovery journal,
changed inventory, or incomplete archive metadata prevents inspection; this
operation does not repair or provision missing source data.

Within that scope, the owning task can call `describe_offline_members()` for
immutable domain/root facts, including original opaque dynamic partition keys
and the archive member. Nested or concurrent entry, initialization and premature
close are rejected. Cancellation settles owned lock acquisition/release first;
failed release retains the remaining locks for the same group's `close()` retry.
Copied descriptors do not retain authorization after the scope exits.

This is a filesystem-specific offline inventory seam, not a portable snapshot
extension. Custom capability partitions still require their dependency contract;
archive membership alone does not prove object closure. Existing snapshot and
opaque-object rejection checks remain in force. It does not retire the filesystem
backend or grant automatic recovery to a SQL source.


## Offline object cleanup and opaque capability state

`Persistence.purge_unreferenced_objects()` requires the caller to quiesce all
users of the routed state and object stores. Its core reference scan cannot
interpret arbitrary capability payloads. A currently bound capability partition
(including a volatile one), or an existing durable capability-presence marker,
therefore returns the selected candidates as `blocked`, before pruning archive
records or deleting objects. Cached maintenance operations observe later
capability bindings. Marker read or integrity failures abort without deletion.
Ordinary core-only cleanup and volatile-state portable snapshot rules are unchanged.

This is not a complete custom-dependency inventory. Raw custom writes that never
published a capability marker cannot be rediscovered reliably after the original
configuration is lost, especially from opaque SQL partition digests. Keep that
source out of generic cleanup until its original owner supplies a complete
reference inventory; do not treat an absent marker as proof of no custom data.
The check adds one source read scope per discovered partition to offline cleanup,
not to Runtime writes or reads. It neither creates markers nor changes schema.
