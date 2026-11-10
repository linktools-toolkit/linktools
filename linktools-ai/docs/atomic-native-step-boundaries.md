# Native turn boundaries

`linktools.ai.capability.steps.AtomicStepPersistence` is the Runtime-bound
adaptation of the public PydanticAI Harness `StepPersistence`. Standalone agents
continue to use the original `StepPersistence`. The adaptation requires a
Runtime-bound store and does not maintain a second standalone tool lifecycle.

The original Agent, RunRecord, StepEvent, ContinuableSnapshot, ModelMessage and
continuation algorithm remain in use. The minimum supported versions are Harness
0.66.0 and PydanticAI 2.66.0. The module reproduces the original MIT attribution.
Upgrades require review of the native node lifecycle, list rebinding and
continuation behavior.
No durable-journal compatibility with another capability implementation is
promised.

## Prepared facts and confirmed publication

Hooks prepare facts for explicit node boundaries. They do not promise an
independently committed write and do not buffer an API with that promise.

- Registration and run-start are prepared at `before_run` and join the first
  model-ready publication. A known-new registration does not first read the
  store. A conflicting insert reads only the registration identity to distinguish
  an idempotent retry from another invocation of the single-shot run ID.
- Model hooks prepare only in-memory facts. Publication is at the public
  model wrapper's `request` or `request_stream` leaf, after capability wrappers,
  the before/prepare hooks and SDK persistent-history finalization. Wrapper
  ordering is resolved after the native CombinedCapability expands all run-level
  and dynamic capabilities, so other model wrappers execute outside the boundary wrapper. The stored context remains the SDK persistent messages,
  not request-only wire rewrites.
- A completed model event joins the response snapshot at the end of its
  ModelRequestNode. It is not deferred until the following tool preflight.
- Tool hooks do not create provisional observations, independent effects or
  finish projections. The actual leaf admission/result owner supplies these
  facts after final argument processing and authorization. Native effect reads
  project that business owner.
- A completed tool node keeps its live context in memory until the next prepared
  context boundary. Known permitted tool outcomes remain independently owned by
  the leaf effect record and are not held hostage by later user hooks.
- Completion combines the native completed event and any necessary final
  snapshot. Failure combines pending native events with the rescued live
  complete/interrupted snapshot before propagating the error.

Only confirmed commits advance local run/snapshot knowledge or emit
`SnapshotSaved`. Notification failure does not retract the committed fact.
Failed or unknown commits invalidate the relevant local read projection;
source readback belongs to the conflict/unknown path, not normal success.

The final model response checkpoint is currently distinct from terminal
publication: the SDK does not know the validated final output until it executes
the following CallToolsNode. The implementation must not silently weaken that
recovery point to meet a transaction budget.

## One existing SQL transaction

`SqlStepStore.prepare_step` captures original records and encodes/externalizes
messages before the transaction. `write_prepared_step(transaction, prepared)`
uses the current writable StateStore transaction's public `run_sql` operation.
It verifies the same AsyncEngine, an open SQL context and current task/scope; it does
not open or commit another transaction. Its body receipt is provisional.

The existing execution writer performs its conditional revision update,
independent execution facts and native writes in that same callback. Only after
that mutation confirms does the caller invoke `confirm_prepared_step`. Callback
failure rolls back every part, including when an inner SQL exception was caught.
This is a SQL composition operation, not a new transaction or checkpoint owner.

`RuntimeStepStore(commit_boundary=...)` delegates a complete prepared boundary
to the execution writer and returns only after confirmation. A caller-owned
native store in another context retains an explicit split-storage commit and
reconciliation path. `SqlStepStore.commit_step` is that native store's own
short transaction and cannot provide same-group execution atomicity. Original
caller-owned stores without `commit_step` publish their individual native writes
separately. A confirmed tool effect is not repeated if subsequent event storage
fails; source readback repairs the keyed event.

A user-defined StepPersistence subclass retains its original hooks and separate
native writes. It does not activate the unified event callbacks. Its lifecycle
observation writes cannot create or settle a business effect; reads project the
actual leaf operation, and explicit annotations require an already admitted
operation. Custom hook execution and native events remain observable. This is an
expensive customization path, not a claim of same-group atomic publication.

The CHECKPOINT control domain must share the Execution storage group under the
current configuration contract. Initialization rejects an independent control
group. This restriction does not prohibit an independent caller-owned native
StepStore; native snapshot storage is a separate configuration boundary.

An uncertain tool invocation preserves business `EFFECT_UNKNOWN` and writes a
native `tool_call_failed` event containing only the safe uncertainty code. When
co-located, the existing tool-fact transaction appends that event after its own
operation fence, without another execution CAS. If native writing rolls back,
the uncertainty fact is still saved and the keyed event is repaired before
reporting confirmed publication. Split stores publish the confirmed effect and
native event separately. Neither path turns uncertainty into a certain failure
or repeats the external effect. Tool events inherit the confirmed RunRecord's
lineage and metadata, with the real leaf step and conversation identity.

## Snapshot identity and retained content

A snapshot header owns its replay identity. A supplied idempotency key is scoped
to the store/run partition; an unkeyed write receives its own occurrence digest.
Pruning leaves the header as a tombstone, marks
it pruned, clears its message-count descriptor and releases its sequence root.
Readers exclude tombstones. Retrying an old key cannot restore its old content
or replace it with different content after pruning or body collection.

Retained snapshots hold an immutable sequence root, not a growing reference list.
The shared sequence forest preserves occurrence order and repetition while
reusing unchanged prefixes. Message bodies are deduplicated by their canonical
singleton-byte SHA-256. Tombstones retain no body references, so they do not become permanent
content-retention roots. The separate snapshot-key table is removed from the
current development schema and its matching reader/tests.

## Cost and reclamation boundaries

Let N be the number of message occurrences, B their total canonical bytes, and
D the new suffix. A confirmed immutable append prepares O(D) new message bodies
and sequence nodes, plus the logarithmic sequence frontier. Frozen message
concatenation reuses the predecessor SHA state and hashes only the appended
bytes; repeated digest requests do not rehash the history. A frozen retry-state
envelope changes only the terminal message and shares the immutable prefix.
These are content operations, not claims about total SQL, model input, or turn
latency.

The SDK exposes mutable history and nested message content to application hooks.
Capturing that arbitrary history still requires O(B) work per boundary to notice
earlier edits. Runtime freezes the logical checkpoint before asynchronous
publication, then converts it to the native representation once and shares the
frozen result through SQL preparation and confirmation. The two representations
can differ for host input descriptors and retry metadata. Object identity or
message count alone is never proof that old content is unchanged. N successive
growing mutable histories can therefore still require quadratic cumulative
capture work; preventing that would require a different SDK mutation contract.

Cold recovery must read and validate the retained history, O(B) in the result
size. It decodes each selected snapshot once and reuses that decoded value for
the returned result and its content proof. Listing every historical snapshot is
proportional to the total requested result, which may itself be quadratic for
successively longer snapshots. Rewrites and cold writers may reconstruct
sequence metadata; they must not assume an in-process committed prefix proof.

Logical retained storage is proportional to unique message bodies, occurrence
nodes and replay headers. Repeated copies of a shared body are not charged as
new content, but each distinct snapshot replay key remains a retained identity.
Keeping all keys and all user-visible history cannot provide constant space.
Configured snapshot pruning retires roots while preserving replay tombstones
and the latest complete recovery point. It does not authorize dropping history
owned by Conversation, Trace, a task handoff, or another retained snapshot.

Native content collection remains an explicitly exclusive, offline operation.
It discovers candidate messages using hashes only, then reads retained bodies
in batches of at most 128 and verifies their full inline/object content before
deleting anything. Orphan bodies are never loaded merely to delete them. Root
and sequence metadata still scale with the namespace; all durable state owners
must be scanned because there is no separate transactional root index. A single
large retained body may exceed the batch's typical memory size. This is bounded
by body count, not an absolute byte cap or a constant-memory collector.

Automatic online deletion is not inferred from an execution reaching terminal
state. Safe online collection requires a retention decision and coordination
with every owner publishing or releasing shared roots, including cached content
proofs. No lifecycle signal bypasses the existing offline exclusion, integrity,
lease, authorization, or unknown-commit contracts.

## Why other storage owners remain

StateStore is the shared CAS/transaction layer for records, aliases, sequences,
facts and operations, not another execution history. Execution owns admission,
principal, producer fences, cancellation, budgets and terminal handoff. Task
owns graph dependencies, scheduling leases and node outcomes across attempts.
Conversation owns cross-run continuation, branches and committed turn order.
None can be recovered reliably from native message snapshots alone.

Trace retains provider requests, partial/failed responses, usage and timing that
are not transcript messages. Shared run/event/message content references its
native owner where available; trace-only and independent storage routes remain
supported. The memory route uses the same state-backed interaction archive as
SQL, rather than maintaining a second private interaction implementation.
Checkpoint control owns tool-effect admission, fencing and unknown outcomes;
native snapshots remain the recovery history owner. History is an authorized
read facade, not an additional durable database.

AssetStore owns logical asset versions, ObjectStore owns immutable large bytes,
Memory owns cross-execution mutable state, and Evaluation and Metrics own their
independent evidence and observations. Those semantics are not removable copies
of StepStore. Large execution bindings already share content-addressed storage;
small bindings stay with their execution row to avoid another read. Their
physical placement does not create a second mutable source of truth.

## Verification

Focused tests cover native continuation and record contents, model-node response
publication, registration replay, Runtime binding, failure rescue, nested lineage,
commit-before-notification, SQL rollback/lost ACK, shared-transaction publication,
no source reads on the known-new boundary, cross-context/scope rejection, pruning,
body collection and replay tombstones. Real MySQL server execution and whole-turn
cost are separate evidence requirements, not inferred from SQLite or SQL compilation.
