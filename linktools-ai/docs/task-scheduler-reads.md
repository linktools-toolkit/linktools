# Scheduler reads

The local launcher distinguishes work that finished before a scheduler read
from work that finishes while that read is in progress. A successful local
task has already completed its runner, heartbeat and execution-hold cleanup.
The next full scheduler read therefore follows its durable writes.

After that read, the launcher removes an already-finished task only when it is
still the same inflight entry, no quiescence has started, and its node remains
present in the returned state. It does not start another quiescence read for
that finished task or repeat reconciliation solely to observe the same completed
retry. Cancelled tasks, failed tasks and quiescence retain their existing paths.
Work that finishes during the read still triggers the existing refresh before
the launcher decides whether to wait or start dependents.

Each scheduler read returns a complete graph state. Mutable node state and
leases remain freshly read, with the same dependency reconciliation and
transactional semantics. Terminal closing reads, cancellation effects, cleanup
failure propagation and execution-hold settlement are unchanged. There is no
new workset protocol, persistent queue or deferred authoritative write.

## Completed local work

`tests/ai/test_task_scheduler_completed_work.py` isolates a local task that has
already completed successfully before the next scheduler read. Another live
node keeps the graph open. Setup, terminal publication and final inspection are
outside the measured interval.

On matched SQLite graphs of 2 and 200 nodes:

- An already-terminal local task needs one scheduler read instead of that read
  plus another quiescence read: SELECT executions fall from 9 to 5.
- A task that has already published a future retry needs one scheduler pass
  instead of two: SELECT executions fall from 10 to 5.
- Both cases use one read-only transaction instead of two. Decoded records fall
  from 10 to 5 at width 2, and from 802 to 401 at width 200. Logical JSON bytes
  are halved in these matched fixtures. Neither case introduces a write.

These are statement counts and decoded logical payload sizes, not network or
disk traffic, and not a whole-graph latency claim. A remaining scheduler pass
still scales with graph size. A task that completes during a read, fails, is
cancelled or is already quiescing does not qualify for this optimization.

An offline Runtime-bound Agent chain was also measured with the same probe on
master `6a2aadb5` and this isolated change, three samples each at widths 2 and 8. The eight-node
samples removed all eight redundant quiescence reads, accounting for 32 SELECT
executions in that attributed path. The execution-hold path remained unchanged.
Whole-chain SELECT medians were 1203 before and 1160 after; checkout medians were
596 and 577. These aggregate
differences include scheduling variation and should not be treated as the exact
32-statement attribution. Write-commit medians remained 132, with one after
sample recording 133; not all other operations are unchanged. The scope included Runtime
attachment, TaskEngine opening, start/wait, all results and closing, excluding
Persistence provisioning. It does not establish real MySQL performance or a
latency improvement.


## Reusing immutable node definitions

A Task repository retains validated definitions for at most its most recently
scheduled active graph. Admission creates immutable definitions; expansion adds
new nodes without changing existing definitions. The scheduler reads the graph
header, all current node-state payloads and lease columns, and the latest event
on every pass. It reuses definitions only for the same graph and the same full
node ID set. A cold read, another graph, or any membership change reads and
validates all definitions alongside that pass's already-read states. A miss does
not repeat the state scan.

Nested transactions neither use nor publish retained definitions: their outer
transaction can still roll back. Only successful top-level mutation or complete
commit readback publishes a definition tuple. Failure and cancellation do not
publish provisional values. Terminal and recovery-required scheduler results
release the retained tuple. A late older read can replace a newer tuple, but the
next fresh membership check reloads definitions if needed. This may cause a
cache miss, not stale mutable state.

The tuple belongs to one repository and source lifetime. Supported Task APIs do
not delete and re-admit a graph ID with new definitions. Snapshot restore opens
a fresh source and repository; it does not replace a live repository's source.
Returned node inputs and nested contract values remain detached through their
existing immutable value interfaces. Retention is bounded to one graph, not a
fixed byte limit: a large active graph can retain large definition payloads until
replaced, observed inactive, or its repository is released.

Public `graph_state`, other full graph reads, and cold recovery still read and
validate complete stored definitions. Each consumed stored payload still
receives its full integrity checks. Direct database or low-level StateStore
modification of an already validated definition is outside the Task writer
contract; a warm scheduler need not detect that modification until a full read.
Missing or corrupt mutable state is not hidden by definition reuse. No SQL status
projection or graph revision is treated as a replacement for authoritative
state or lease payloads.

`tests/ai/test_task_scheduler_definitions.py` protects these boundaries across
memory, local SQLite and SQL SQLite. Its controlled SQLite budgets include all
current state payloads, not just saved definition reads. At widths 2 and 200, an
unchanged warm pass reads respectively 3 and 201 records instead of 5 and 401,
and executes 4 SELECTs instead of 5, in one transaction without writes. Cold,
nested and membership-miss passes retain the full definition-read cost; changing
graphs deliberately evicts the previous graph. These are record and SQL budgets,
not network traffic or whole-chain latency guarantees. The scheduler still does
O(N) state reading and graph validation.

## Target-node settlement reads

Execution-hold cleanup, completion readback, and waiting-recovery checks consume
only the selected node's current state. They use the existing `get_node_states`
operation to read that node's state and definition in a fresh read transaction.
They do not reuse a scheduler snapshot or retain mutable state between calls.
Execution identity, result digest, fence, terminal status, cancellation shielding
and cleanup failure propagation retain their existing checks. A recovery write
still applies the repository's original atomic transition and fence validation.

These reads validate the consumed target records, including missing or malformed
state and definitions. They are not whole-graph integrity scans: unrelated
records, graph headers, summaries and event streams remain the responsibility of
full graph readers and writers. Supported Task operations do not delete a graph
header while retaining live nodes. Low-level damage outside the target records
is not detected by this local read. Public `graph_state` and terminal closing
reads remain complete and fresh.

On controlled SQLite graphs of 2 and 200 nodes, a terminal hold/readback check
reads 2 records and executes 1 SELECT, instead of reading 5 or 401 records and
executing 4 SELECTs. Both versions use one read transaction and commit, without
writes. These budgets exclude the producer-owned hold-release operation; they
are not whole-operation transaction or latency claims.
