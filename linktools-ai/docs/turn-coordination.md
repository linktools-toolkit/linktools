# Live turn coordination

A host may expose the optional `TurnCoordinationBackend` method group on the
object passed explicitly as `PersistenceStorage.sqlite(path, coordination=backend)`.
`cache=...` remains a separate hot-read resource; it does not enable coordination. The group is
independent of hot bytes, pending writes, and preview streams. Adapters need
only the complete groups they use; a partial group is a configuration error.
Runtime and independent History borrow the same configured resource through
Persistence; there is no second Runtime coordinator argument or required base class.

```python
from linktools.ai.storage import Lease, TurnCoordination, TurnCoordinationBackend

async def claim_turn(
    scope: str, *, session_key: str, request_key: str, request_digest: str,
    execution_id: str, owner_id: str, lease_seconds: float,
) -> TurnCoordination | None: ...

async def read_turn(scope: str, execution_id: str) -> TurnCoordination | None: ...
async def renew_turn(lease: Lease, *, lease_seconds: float) -> TurnCoordination | None: ...
async def release_turn(lease: Lease) -> bool: ...
async def cancel_turn(scope: str, execution_id: str) -> bool: ...
```

`TurnCoordination` is frozen and contains `execution_id`, `owner_id`,
`cancel_requested`, and an optional existing `Lease`. A newly acquired claim
contains its lease. A successful renewal returns that same lease plus current
cancellation state. Duplicate claims and reads contain no lease, so observing
an execution never grants ownership. The backend interprets `Lease.key`;
callers do not parse it or share a backend-private byte codec.

## Identity and atomicity

The caller supplies one opaque scope containing both tenant identity and the
Runtime namespace. Participants must use the existing same-source namespace
contract: shared coordination is meaningful only when they share the same
authoritative storage. Separate sources must not accidentally share a scope.

Session, request, and execution keys are isolated by scope. Admission atomically
checks a request and claims its session, so two distinct requests cannot both
own the same live session. A duplicate request returns the original execution
and owner, regardless of the proposed new execution and owner IDs. Reusing a
live request key with a different digest or session raises
`AIError(ErrorCode.IDEMPOTENCY_CONFLICT)`. A busy session returns `None`.
Reassigning a live execution ID to another request raises `STORAGE_CONFLICT`.

Every new generation supplies a fresh execution ID and owner ID. Execution IDs
must never be reused after release or expiry: cancellation is addressed by
execution ID, and a late cancellation of an older generation must not target
a newer one. `owner_id` identifies this acquisition, not a stable process name.
All input strings must be nonempty; lease duration must be a finite positive
number, excluding booleans.

## Lifetime and failures

Reads, duplicate claims, and cancellation do not extend expiry. Only a matching
live lease can renew or release ownership. Renewal returns `None` for a lost
lease; release returns `False` for stale, expired, or absent ownership. An old
lease's token or fence cannot affect a newer owner of the same session.

Cancellation sets the live execution's flag without releasing its session or
renewing expiry. Repeated cancellation returns `True` while that execution is
live; absent executions return `False`. A renewing owner observes the flag.

Released or expired claims no longer provide replay authority. Completed
request results, recovery, and durable state still belong to authoritative
storage. Shared coordination alone does not provide durable execution history
or permission for an expired worker to commit: the owning Runtime must enforce
its existing fencing and durable transition checks.

Errors, including outages and ambiguous mutation outcomes, propagate. Consumers
must not use best-effort hot-content helpers to convert coordination errors into
misses or start a replacement execution based on an unknown claim outcome.
The SDK defines this protocol without a Redis dependency, key encoding, or
connection lifecycle. The host owns its client and distributed implementation.

## Built-in in-memory support

`InMemoryCacheBackend` supplies this group alongside its hot-content and pending
write groups. Coordination is shared only by consumers of that exact cache
instance in one event loop and process. Separate instances, processes, and pods
do not coordinate with each other.

Live ownership metadata is separate from `max_bytes` and `max_size`: content
pressure and pending writes never evict it. Metadata is reclaimed after lease
expiry on coordination calls, or immediately on successful release. No task,
connection, or background timer is created by this capability.
