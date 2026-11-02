# Atomic writes from capabilities

A Runtime-scoped native capability can opt into `CapabilityPersistence` from
`linktools.ai.persistence`. It is a structural protocol, not a required
base class. The capability declares `commit`; physical resources belong only to
`PersistenceStorage.plan`. By default each capability gets an isolated partition
on the execution route, without a second registration. Register its `id` and
`revision` through `RuntimeFeatures.capability`.

```python
from collections.abc import Mapping
from copy import copy
from hashlib import sha256

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext

from linktools.ai.capability import AgentContext
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import CommitPolicy, JsonValue, canonical_json_bytes
from linktools.ai.persistence.capabilities import CapabilityWriter
from linktools.ai.persistence.state import StateTransaction, StoredRecord
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime

class RunNote(AbstractCapability[AgentContext[None]]):
    def __init__(self, *, commit: CommitPolicy) -> None:
        self.commit = commit
        self.writer: CapabilityWriter | None = None

    def bind_persistence(self, writer: CapabilityWriter) -> "RunNote":
        bound = copy(self)
        bound.writer = writer
        return bound

    async def before_run(self, ctx: RunContext[AgentContext[None]]) -> None:
        assert self.writer is not None
        await self.writer.submit(
            key="started", payload={"execution_id": ctx.deps.execution_id},
        )

    async def persist_in_transaction(
        self, transaction: StateTransaction, payload: Mapping[str, JsonValue],
    ) -> None:
        identity = sha256(canonical_json_bytes(dict(payload))).digest()
        await transaction.insert_record(StoredRecord(
            key_digest=identity, scope_digest=identity, parent_digest=None,
            kind="run-note", sort_key=identity.hex(), state=None,
            storage_revision=0, lease_owner=None, lease_fence=0,
            lease_expires_at=None, data=payload,
        ))

features = RuntimeFeatures("notes")
features.capability(
    RunNote(commit=CommitPolicy.terminal()),
    id="application.run-note", revision=1,
)
# Register models on features before running agents.
async with Persistence.open("notes", storage=PersistenceStorage.in_memory()) as persistence:
    async with Runtime.attach(persistence=persistence, features=features) as runtime:
        ...
```

Different participants use separate domain partitions. Ordinary write scopes
coalesce when they resolve to the same transaction resource in the same Runtime
and run, with equal commit values and the same borrowed buffer instance. Policy
object identity does not matter. At terminal preparation, all terminal scopes
for that execution/run on the terminal transaction resource contribute their
prefixes, even when they have different buffers. Each buffer still owns its own
entries and acknowledgement IDs. A failed domain callback rolls back the whole
combined source transaction and retains the prefixes; it does not partially
commit one participant. Default Metrics writes keep their independent best-effort
contract.

Custom placement is optional: set `PersistencePlan.capability_routes` using the
registered capability ID. `trace` and `checkpoint` similarly configure archive
resources. Runtime features never override the shared physical plan. Multiple
Runtimes may borrow one Persistence while keeping distinct callback queues,
staging, revisions and close boundaries. The host closes Persistence after all
producers; closing one Runtime never closes another producer's shared resources.

## Commit boundaries

- `sync()` confirms the submitted write before returning. Sequential synchronous
  calls are not delayed to invent a larger batch.
- `batch(batch_size=..., flush_interval=..., buffer=...)` aggregates accepted
  entries across compatible participants. The first accepted entry starts the
  deadline. A full buffer rejects a submission rather than waiting while holding
  the flush lock; callers may flush and retry. Without a buffer, batch writes
  synchronously confirm, as with existing policy behavior.
- `terminal(buffer=None)` defers until the existing execution terminal
  transaction. Without a buffer it keeps existing local staging; with a buffer,
  each accepted entry contains immutable serialized payload bytes. All compatible
  participant scopes contribute their selected prefixes, including on the
  pending-turn path. Other physical resources cannot join that transaction.
- For these explicit capability writers, `writer.flush()` and Runtime close
  are forced boundaries. They can persist a terminal-policy prefix independently
  of execution completion,
  including after a failed execution terminal attempt. Terminal policy is a
  scheduling boundary, not a promise that every stored note proves success.

A successful deferred `submit` means accepted, not durable. Buffered batch and
terminal writers read back and validate the whole fixed selected prefix before
calling the source callbacks; a short/missing/corrupt read is not permission to
submit a RAM copy. Capacity rejection uses `wait=False`, not a wait that depends
on the same execution reaching its terminal boundary. For a buffered batch or
terminal writer, an unhandled known rejection fails the turn without repeating a
provider request or silently changing the policy to synchronous SQL; required
Execution control still has to settle. Unknown append/source outcomes keep their
original IDs for readback.

After a failed flush, retry explicitly; no background retry loop is created.
Close propagates failure and retains unresolved work. Canonical terminal
confirmation and release of owned source locks precede buffer acknowledgement. An acknowledgement failure cannot turn that confirmed business
result into failure; it retains cleanup for retry without repeating the source
transaction. Explicit flush/close can still report outstanding cleanup. Borrowed
buffers, engines and caches remain caller-owned. Retrying persistence
does not erase a recorded execution-worker failure: Runtime close may continue
to report that failure even after the retained writes are confirmed.

The framework stores idempotency receipts alongside domain changes. Keys are
scoped by capability ID, revision and run. Replaying an identical key/payload
avoids repeating the domain operation; changing a payload for that key conflicts.
Payloads are frozen JSON values at submission. Use existing transaction CAS
operations for mutable records, and respect record ownership when inserting
facts. Domain callbacks must perform only supported transaction-store operations,
with no network calls or other external side effects. Commit uncertainty uses the
existing durable readback mechanism, not callback retries on guessed outcomes.

## Scope and recovery limits

Bindings must return independent run-local instances and preserve commit settings. A sealed execution rejects later writes or new bindings.
This initial protocol supports Runtime-scoped declarations only; selecting it on
an Agent-scoped declaration is rejected. Ordinary Agent capabilities are
unaffected.

Payload bytes live in the borrowed buffer, but callback bindings remain
owner-local. This adds no query API or pending-read view, and does not promise
RAM unloading. A shared buffer is not a durable task queue
and does not enable another Pod to execute Python callbacks after owner loss.
Already committed records remain in the configured backend; uncommitted memory
is subject to the selected deferred policy's loss boundary. Execution recovery
and checkpoint safety boundaries do not move into this optional queue.

Portable snapshot export is rejected when durable custom partitions are configured.
Memory and transient custom partitions follow the existing volatile-domain policy:
they are omitted from portable snapshots and do not establish a sticky marker.
A Runtime with both volatile and durable custom partitions still rejects export.
A sticky presence marker in an existing durable core store also prevents export
after durable custom data has committed, even if the declarations are omitted on reopen.
Filesystem export also checks existing transaction-group membership before opening
a reduced resource set. Read-only History borrows Persistence without loading capability callbacks.
Reopen with the same physical plan for explicitly routed external resources;
default partitions are discovered inside their existing group.
The first durable custom transaction includes that marker; a separate core resource
requires a confirmed marker transaction before the custom transaction starts.
Once confirmed, the Runtime does not repeat marker operations for later writes.
The marker conservatively continues to block export; this is not a partition
registry or a portable custom-data format. JSON payloads do not register Asset or
Object ownership, retention references, or GC roots. Store self-contained domain
values; do not rely on an embedded object ID to retain an external asset.

Only supported transaction-store changes receive that adapter's publication and
cache behavior. This protocol does not publish arbitrary application hot state.
One transaction may still execute several SQL statements and network round trips.
