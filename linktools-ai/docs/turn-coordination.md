# Live turn coordination

A host may expose the optional `TurnCoordinationBackend` method group on the
object passed explicitly as `PersistenceStorage.sqlite(path, coordination=backend)`.
`cache=...` remains a separate hot-read resource; it does not enable coordination. The group is
independent of hot bytes, pending writes, and preview streams. Adapters need
only the complete groups they use; a partial group is a configuration error.
Runtime and independent History borrow the same configured resource through
Persistence; there is no second Runtime coordinator argument or required base class.

```python
from linktools.ai.storage import Lease, TurnCoordination, TurnCoordinationBackend, TurnScope

def enter(
    scope: str, *, session_key: str, request_key: str, request_digest: str,
    execution_id: str, owner_id: str,
) -> TurnScope: ...

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

## Owner lifetime

The six-method coordination group is one capability. `enter` creates a
single-use `TurnScope`; the host implements lease policy, renewal and signals.
Runtime enters and exits the scope in the **same owner-lifetime task**. It does
not transfer an asynchronous context manager between the admission caller and
worker tasks, so context variables and task groups retain their normal meaning.

```python
class TurnScope(Protocol):
    async def __aenter__(self) -> TurnCoordination | None: ...
    def require_active(self) -> None: ...
    async def wait_signal(self) -> Literal["owner_lost", "cancel_requested"]: ...
    async def __aexit__(self, exc_type, exc, traceback) -> None: ...
```

Entry preserves the primitive's three outcomes: busy (`None`), replay
(observation without a lease), and ownership (with a lease). Only ownership
permits execution, local validity checks and signal waits. `require_active`
is synchronous and performs no network IO. Once lost, a scope never becomes
valid again. Signal waits must wake on the local deadline even when renewal
is blocked. Backend errors propagate rather than becoming contention or absence.

`cancel_requested` is delivered once; further waits can still report
`owner_lost`. Cancellation does not stop renewal or release ownership. Runtime
maps cancellation to its existing source cancellation workflow and maps owner
loss to stopping this exact worker generation. Initial claim cancellation is
also delivered. A lease is not proof that source cancellation has not occurred.

Runtime keeps the scope across model and tool calls, checkpoints, child waits
and durability promotion. Returning a handle, cancelling the caller or timing
out a wait does not release it. Before exit, Runtime stops new dispatch and
settles the runner, admitted child starts, old child workers, entered tools and
owned durable work. Exit settles renewal and releases only its captured lease.
Release failure cannot replace an already established business result.

A backend can reuse `RenewingTurnScope` or supply its own implementation:

```python
from linktools.ai.storage import RenewingTurnScope

# Method on the host's complete coordination backend:
def enter(self, scope, **identity):
    return RenewingTurnScope(self, scope, **identity,
                             lease_seconds=30, renew_seconds=5)
```

The host owns these policy values, its distributed client and worker routing.
Runtime does not run a second renewal loop or schedule replacement workers.
`InMemoryCacheBackend` and `StateTurnCoordination` provide this method. A host
backend need not expose `CacheBackend`; without cache support it still owns the
whole turn, but has no pending-view optimization.

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

An execution ID identifies one logical turn. Recovery retains that execution ID
and its original session/request identity, but supplies a fresh `owner_id` for
each ownership generation. An expired or released turn can be claimed by its
new owner; a live turn cannot be preempted. A different logical turn must use a
different execution ID. `owner_id` identifies this acquisition, not a stable
process name.
All input strings must be nonempty; lease duration must be a finite positive
number, excluding booleans.

## Lifetime and failures

Reads, duplicate claims, and cancellation do not extend expiry. Only a matching
live lease can renew or release ownership. Renewal returns `None` for a lost
lease; release returns `False` for stale, expired, or absent ownership. An old
lease's token or fence cannot affect a newer owner of the same session.

Cancellation addresses the logical execution, so a delayed cancellation also
applies to its current recovered owner. It sets the live execution's flag
without releasing its session or renewing expiry. Repeated cancellation returns
`True` while that execution is live; absent executions return `False`. A renewing
owner observes the flag. Generation-specific renewal and release still require
the exact lease and cannot affect a replacement owner.

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

## Parents, recovery and the source boundary

Parent and child executions have independent owners. A parent's scope remains
live while it waits; siblings do not serialize behind a parent/root execution
lock. Runtime tracks exact active child scopes separately from the delegations
authorized by that parent. A deferred child's worker can stop while its live
parent retains permission for an approval or external result to continue it.
A cold scan does not grant that permission. A replacement parent must explicitly
replay the original invocation; completed results remain reusable without
restarting work, and cancelled children are not reopened.

Owner loss closes the parent gate before stopping its old workers. It does not
force-cancel logical children. Entered external effects settle before release;
unknown effects retain the existing recovery requirements and are not blindly
retried. Without restart checkpoints, an interrupted owned worker is recorded
as `FAILED`, not business `CANCELLED`, and no automatic replay is promised.
With checkpoints, a safely stopped worker retains its recoverable state.

New-child reservation checks the captured parent producer generation and
`STARTED` status in the same source transaction that creates the child. The
child's own generation CAS remains the write fence. The lifecycle gate covers
all producer installation paths, including replay and deferred recovery; it
does not add a parent source read to every model, checkpoint or resume command.

Coordination and source storage are not one transaction. Propagation delay
exists between lease loss and its observation; neither this protocol nor an
extra parent lease can promise instantaneous cross-store fencing or exactly-once
third-party effects. Public Runtime cancellation signals the logical turn after
its source cancellation or recovery cancellation intent is committed. A remote
force cancellation may settle the source first; its owner still drains the
local worker when the signal arrives. Delivery can fail or the caller process
can exit between source commit and notification, so durable model/tool control
boundaries retain fresh source reads even while the scope is valid.

Without coordination, Runtime still tracks local lifetimes, uses fresh source
control reads and keeps source generation CAS. It does not silently create a
distributed lock or promise cross-worker lifetime exclusion.

## Built-in in-memory support

`InMemoryCacheBackend` supplies this group alongside its hot-content and pending
write groups. Coordination is shared only by consumers of that exact cache
instance in one event loop and process. Separate instances, processes, and pods
do not coordinate with each other.

Live ownership metadata is separate from `max_bytes` and `max_size`: content
pressure and pending writes never evict it. Metadata is reclaimed after lease
expiry on coordination calls, or immediately on successful release. The atomic
primitives create no tasks or connections. An entered `RenewingTurnScope` owns
its renewal task and deadline timer until exit.

## Read-only admission lookup

A coordination backend may also implement the independent `TurnAdmissionReader`
capability for indexed request lookup:

```python
async def read_turn_by_request(scope: str, request_key: str) -> TurnCoordination | None: ...
```

This reads the request index already maintained by `claim_turn`. It returns the
live owner's token-free identity, or `None` when no live claim exists. It does
not claim, renew, release, or persist anything. The same scope isolation,
nonempty key validation, failure propagation, and expiry rules apply.

The capability is optional: existing coordination backends remain usable for
execution without it. Admission lookup on such a backend fails explicitly with
`REQUEST_FIELD_INVALID` and reason `admission_lookup_not_supported`, rather than
silently disabling coordination or treating the request as absent.

A live claim can precede publication of its readable execution view. The
persistence admission reader reports `EXECUTION_START_UNKNOWN` if the claim
exists but its view is missing, invalid, or loses ownership while being read.
Callers must not interpret that outcome as permission to start with a new key.
`InMemoryCacheBackend` supports both capability groups.

## Turn context and storage publication

The active worker reuses its captured binding, input context, message frontier
and confirmed checkpoint prefix. State records remain source-authoritative and
revisioned; immutable archive content is addressed by digest and sequence.
Shared content caches and in-process prepared data are not additional control
authorities. Cold recovery reads the required source checkpoint and receipts.

Confirmed checkpoint receipts retain immutable archive nodes for subsequent
checkpoint and interaction preparation. Only new publication records enter the
write set. A failed or unresolved publication does not advance that receipt,
and the existing run-cache release discards it. Publication dependency work
follows the new records rather than scanning the entire cached history.

The bound native same-group path advances one Execution control fence from its
confirmed storage revision, without a normal-path control SELECT or a second
history-head owner. Its short transaction includes the prepared native records;
confirmation advances the local receipt. Unknown or conflicting writes return
to the source. Initial execution input and producer admission share the existing
prepared-attempt transaction when eligible. Other dispatch paths still use fresh
source control. Local lease validity alone never proves that business
cancellation has not occurred.

Terminal result publication and post-consumer retention release have different
lifetimes. They are not merged before consumers and dependency holders have
finished. External tool intent/outcome and unknown-commit reconciliation remain
independent durable facts, even when adjacent metadata reads are combined.
