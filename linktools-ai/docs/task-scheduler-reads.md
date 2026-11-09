# Scheduler reads after local completion

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

Each scheduler read remains a fresh, complete definition/state read with the
same integrity checks, dependency reconciliation and transactional semantics.
Terminal closing reads, cancellation effects, cleanup failure propagation and
execution-hold settlement are unchanged. There is no cross-read cache, new
workset protocol, persistent queue or deferred authoritative write.

## Measured boundary

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
