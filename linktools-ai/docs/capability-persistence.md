# Atomic writes from capabilities

A Runtime-scoped native capability can opt into `CapabilityPersistence` from
`linktools.ai.runtime.persistence`. It is a structural protocol, not a required
base class. Configuration remains `storage` and `commit`; `storage=None` uses
that Runtime's execution route. Register an explicit capability `id` and
`revision` through `RuntimeFeatures.capability`.

```python
from collections.abc import Mapping
from copy import copy
from hashlib import sha256

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext

from linktools.ai.capability import AgentContext, RuntimeFeatures
from linktools.ai.core import CommitPolicy, JsonValue, canonical_json_bytes
from linktools.ai.runtime.persistence import CapabilityWriter
from linktools.ai.runtime.state import StateTransaction, StoredRecord
from linktools.ai.storage import StorageRoute

class RunNote(AbstractCapability[AgentContext[None]]):
    def __init__(self, *, storage: StorageRoute | None, commit: CommitPolicy) -> None:
        self.storage = storage
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
    RunNote(storage=None, commit=CommitPolicy.terminal()),
    id="application.run-note", revision=1,
)
# Supply features, models and RuntimeStorage to Runtime.open as usual.
```

Two different participants use separate domain partitions but can share one
physical transaction. They must resolve to the same transaction resource in the
same Runtime, belong to the same run, and have equal commit values and the same
borrowed write buffer. Policy object identity does not matter. Participating in
this protocol selects atomic failure handling: a failed domain operation rolls
back the whole compatible prefix, retains it and raises. Default Metrics writes
keep their independent best-effort contract.

## Commit boundaries

- `sync()` confirms the submitted write before returning. Sequential synchronous
  calls are not delayed to invent a larger batch.
- `batch(batch_size=..., flush_interval=..., buffer=...)` aggregates accepted
  entries across compatible participants. The first accepted entry starts the
  deadline. A full buffer rejects a submission rather than waiting while holding
  the flush lock; callers may flush and retry. Without a buffer, batch writes
  synchronously confirm, as with existing policy behavior.
- `terminal()` defers until the existing execution terminal transaction. A
  compatible tail joins that transaction, including the pending-conversation
  path. Other physical resources cannot join it.
- `writer.flush()` and Runtime close are explicit forced boundaries. They can
  persist a terminal-policy prefix independently of execution completion,
  including after a failed execution terminal attempt. Terminal policy is a
  scheduling boundary, not a promise that every stored note proves success.

A successful deferred `submit` means accepted, not durable. After a failed
flush, retry explicitly; no background retry loop is created. Close propagates
failure and retains unresolved work. Buffer acknowledgement follows confirmed
commit; acknowledgement failure retries acknowledgement without repeating the
transaction. Borrowed buffers, engines and caches remain caller-owned. Retrying persistence
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

Bindings must return independent run-local instances and preserve storage and
commit settings. A sealed execution rejects later writes or new bindings.
This initial protocol supports Runtime-scoped declarations only; selecting it on
an Agent-scoped declaration is rejected. Ordinary Agent capabilities are
unaffected.

Pending callbacks are owner-local. A shared buffer is not a durable task queue
and does not enable another Pod to execute Python callbacks after owner loss.
Already committed records remain in the configured backend; uncommitted memory
is subject to the selected deferred policy's loss boundary. Execution recovery
and checkpoint safety boundaries do not move into this optional queue.

Portable snapshot export is rejected when these custom partitions are configured.
A sticky presence marker in an existing durable core store also prevents export
after custom data has committed, even if the declarations are omitted on reopen.
Filesystem export also checks existing transaction-group membership before opening
a reduced resource set. Read-only history must reopen with the same declarations
to materialize the original filesystem group.
The first compatible transaction includes that marker; a separate core resource
requires a confirmed marker transaction before the custom transaction starts.
Once confirmed, the Runtime does not repeat marker operations for later writes.
The marker conservatively continues to block export; this is not a partition
registry or a portable custom-data format. JSON payloads do not register Asset or
Object ownership, retention references, or GC roots. Store self-contained domain
values; do not rely on an embedded object ID to retain an external asset.

Only supported transaction-store changes receive that adapter's publication and
cache behavior. This protocol does not publish arbitrary application hot state.
One transaction may still execute several SQL statements and network round trips.
