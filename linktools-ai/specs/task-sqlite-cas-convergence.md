# TaskGraph SQLite Optimistic Concurrency Convergence Spec

Status: Final revised

Baseline: `e1c8592230f5bcd48aee2c7db38064d8b0b7fca6`

This spec is subordinate to repository `AGENTS.md` and `linktools-ai/AGENTS.md`.

This change does **not** change the package version, persistence format, database schema, or public Runtime API.

## 1. Problem

When `RuntimeDomain.TASK` is routed to a durable SQLite `StateStore`, the built-in TaskGraph scheduler, recovery path, node runners, heartbeats, cancellation, and waiters can concurrently observe or mutate the same TaskGraph/TaskNode aggregate.

SQLite makes the race easy to reproduce because writers serialize, but the correctness defect is backend-neutral: Runtime semantic CAS conflicts are not consistently converged by the Task domain that owns the state-machine invariant.

The SQL layer already distinguishes database-native retryable aborts from semantic CAS conflicts. That separation must remain:

```text
database retryable abort
    -> SqlStorageContext retry

semantic STORAGE_CONFLICT
    -> Task repository exits the failed transaction
    -> rereads durable Task state
    -> recomputes the semantic decision
    -> retries only when the latest state still permits it
```

`STORAGE_CONFLICT` must not become a generic SQL retry condition.

## 2. Goals

The implementation must:

1. make SQLite-backed TaskGraph execution stable for `max_concurrency == 1` and `> 1`;
2. prevent normal scheduler/recovery/heartbeat/terminal CAS races from leaking `STORAGE_CONFLICT`;
3. prevent those normal races from being converted into `STORAGE_RECOVERY_REQUIRED` or persisted node failure;
4. preserve crash/reopen TaskGraph recovery;
5. preserve real ownership, fencing, idempotency, integrity, and storage errors;
6. make `inspect_graph()` and `wait_graph()` read-only observers;
7. keep TaskNode durable state authoritative and graph/recovery status derived;
8. avoid pessimistic locks, hidden global locks, new coordination records, schema changes, and public API expansion.

## 3. Non-goals

Do not add or change:

- `SELECT ... FOR UPDATE`, advisory locks, graph-level/global mutexes, or SQLite-specific locks;
- WAL or `busy_timeout` as a semantic correctness fix;
- generic retry of `STORAGE_CONFLICT` in SQL storage;
- Task filesystem fallback;
- Task coordination tables/records;
- `Runtime.open(...)` launcher injection;
- SQLite-specific Task launcher/configuration;
- database schema or durable wire format;
- package version;
- ToolOperation behavior;
- unrelated Runtime-domain concurrency abstractions.

Do not modify `storage/_database.py`, `storage/_dialects.py`, or `runtime/state/_sql.py` for this change.

## 4. Core invariants

### 4.1 Transaction ownership

A transaction helper performs exactly one mutation attempt. It never retries by recursively entering `StateStore.mutate()`.

The method that owns the complete atomic invariant owns semantic retry. A retry is valid only after the previous transaction exits and all decision state is reread.

```text
transaction A
  -> read
  -> compute
  -> CAS conflict
  -> rollback / exit

transaction B
  -> fresh read
  -> recompute
  -> retry only if latest semantic state permits it
```

Never reuse stale records, `storage_version`, or mutation candidates across attempts.

### 4.2 Task authority

TaskNode records are authoritative for Task execution state.

`task_graph.status` and the Task admission recovery state are derived projections/indexes:

```text
TaskNode states
    -> task_graph.status
    -> task_admission recovery projection
```

Graph topology remains immutable.

### 4.3 Lease/fence semantics

Heartbeat failure never reacquires ownership automatically.

Terminal retry is allowed only when a fresh read proves the node is still `RUNNING`, owned by the same owner/fence, and the lease is still live. The retry persists only the terminal state; it never reruns node business execution.

### 4.4 Observer purity

Read APIs must not perform state advancement merely because they inspect or wait for a graph.

State advancement belongs to scheduler/recovery/cancellation writers.

## 5. Production changes

Primary files:

- `linktools-ai/src/linktools/ai/runtime/state/_task_recovery_repository.py`
- `linktools-ai/src/linktools/ai/task/_service_impl.py`

The base Task repository in `_repositories.py` remains the one-attempt semantic implementation. The durable materialized repository owns the outer convergence loop because its invariant includes recovery-projection synchronization.

`linktools-ai/src/linktools/ai/task/_local.py` must not broadly swallow repository conflicts. If expected Task CAS still reaches the launcher, repository convergence is incomplete.

## 6. DurableTaskRepository convergence

### 6.1 `reconcile_graph()`

`DurableTaskRepositoryImpl.reconcile_graph()` owns the complete retry unit:

```text
fresh StateStore mutation
  -> read graph and all node rows
  -> recompute READY/BLOCKED node transitions
  -> recompute graph status
  -> atomically persist required node/graph changes
  -> synchronize recovery projection
```

On `STORAGE_CONFLICT`:

1. exit the failed transaction;
2. yield only for event-loop fairness if needed;
3. start a new mutation transaction;
4. reread graph and all nodes;
5. recompute from the new state.

All non-`STORAGE_CONFLICT` errors propagate unchanged.

Missing dependency rows are `STORAGE_INTEGRITY_ERROR`, not raw `KeyError`.

### 6.2 `cancel_graph()`

Cancellation follows the same fresh-transaction convergence rule.

For every retry, reread all nodes and cancel only nodes that are still non-terminal.

Required race semantics:

- completion commits `SUCCEEDED` first -> cancellation preserves that node and cancels remaining non-terminal nodes;
- cancellation commits `CANCELLED` first -> an old terminal writer fails fresh lease validation with `TASK_FENCE_STALE`;
- terminal nodes are never rewritten by cancellation.

The graph projection and recovery projection must converge to the status derived from the resulting nodes.

### 6.3 `claim()`

A raw storage-version CAS miss starts a fresh `claim()` attempt.

The fresh attempt must reclassify the latest state into existing semantic outcomes, for example:

- return a new `TaskLease` when the node is still claimable;
- `TASK_OWNER_CONFLICT` when another live owner won;
- `TASK_NOT_READY` when state/dependencies no longer allow claiming;
- existing integrity/storage errors for real faults.

Normal claim competition must not leak raw `STORAGE_CONFLICT`.

### 6.4 `renew()`

Do not change current fencing behavior.

A heartbeat CAS miss remains `TASK_FENCE_STALE`. Do not automatically retry or reacquire a lease.

### 6.5 `complete()` / `fail()`

A terminal CAS miss may be caused by a same-owner/same-fence heartbeat advancing `storage_version` between the terminal read and write.

On `STORAGE_CONFLICT`, start a fresh terminal persistence attempt. The base lease validation then decides whether retry remains valid.

If the latest node is still:

- `RUNNING`;
- same owner;
- same fence;
- live lease;

then retry terminal persistence using the latest storage version.

If status, owner, fence, or lease validity changed, return `TASK_FENCE_STALE`.

Never replay `TaskNodeRunner.run()`.

### 6.6 Retry policy

Do not use a fixed `MAX_CAS_RETRIES` for normal semantic convergence.

Each retry must be a fresh durable observation. `await asyncio.sleep(0)` may be used only for event-loop fairness. Do not add fixed or exponential sleeps inside the semantic convergence loop.

Caller cancellation remains able to interrupt the loop.

## 7. Task service behavior

### 7.1 `inspect_graph()` is read-only

After authorization, `DefaultTaskService.inspect_graph()` must call `tasks.get_graph()`.

It must not call `reconcile_graph()`.

If the authorized graph header existed but the subsequent graph read unexpectedly returns `None`, return `STORAGE_INTEGRITY_ERROR`.

### 7.2 `wait_graph()` is read-only

Polling must call `tasks.get_graph()`.

Scheduler-failure readback must also call `tasks.get_graph()`.

The waiter must not advance node state, graph projection, or recovery projection.

If the graph disappears after successful authorization, return `STORAGE_INTEGRITY_ERROR`.

### 7.3 `recover_pending()` remains a writer

Do **not** replace recovery reconciliation with a pure read.

A process can crash after TaskNode terminal state is durable but before `task_graph.status` and the recovery index are refreshed. Reopen recovery must call `reconcile_graph()` so stale non-terminal projections are repaired and terminal graphs disappear from recoverable enumeration.

## 8. Error contract

Normal internal CAS competition converges internally and is reread/reclassified.

The following remain externally observable when semantically real:

| Condition | Error |
| --- | --- |
| another live task owner wins | `TASK_OWNER_CONFLICT` |
| stale/lost lease or fence | `TASK_FENCE_STALE` |
| task is no longer claimable | `TASK_NOT_READY` |
| incompatible duplicate request | existing idempotency/conflict error |
| corrupt durable Task state | `STORAGE_INTEGRITY_ERROR` |
| unavailable storage | `STORAGE_UNAVAILABLE` |
| unresolved commit outcome | existing `STORAGE_COMMIT_UNKNOWN` contract |

A normal scheduler/waiter/heartbeat race must not be persisted as a node `FAILED(error_code=STORAGE_CONFLICT)`.

## 9. I/O and concurrency constraints

The uncontended path must not gain an additional transaction.

For a genuine semantic CAS race, each retry uses one new `StateStore.mutate()` transaction.

Do not:

- sleep while holding a storage transaction;
- hold a lock across `TaskNodeRunner.run()`;
- introduce a background retry worker;
- create new durable records only for coordination;
- add process-global coordination.

`inspect_graph()` and `wait_graph()` should reduce SQLite write pressure because they become read-only.

## 10. Tests

Tests are mandatory for this change.

### 10.1 Deterministic repository race tests

Cover at least:

1. `reconcile_graph()` CAS conflict -> fresh reread -> convergence;
2. `cancel_graph()` racing node completion -> first committed terminal fact is preserved;
3. `claim()` CAS conflict -> fresh semantic reclassification;
4. `complete()` racing same-owner/same-fence heartbeat -> terminal commit succeeds after fresh reread;
5. `fail()` racing same-owner/same-fence heartbeat -> terminal commit succeeds after fresh reread;
6. terminal retry after cancellation/new fence -> `TASK_FENCE_STALE`;
7. corrupted/missing dependency -> `STORAGE_INTEGRITY_ERROR`, no infinite retry.

Tests must assert durable state, not only retry counts.

### 10.2 Observer purity tests

For `inspect_graph()` and `wait_graph()` when no scheduler progress occurs, assert observation does not mutate:

- node `storage_version`;
- graph `storage_version`;
- lease owner;
- fence;
- lease expiry;
- recovery projection.

### 10.3 SQLite public Runtime tests

Exercise the public composition path using `RuntimeStateRoute.sqlite(path)` / public Runtime open/run APIs.

Required scenarios:

- single node, `max_concurrency=1`;
- dependency graph `A -> B`, `max_concurrency=1`;
- parallel independent nodes with `max_concurrency>1`;
- dependency chain plus parallel branch;
- node failure and dependent blocking;
- cancellation;
- waiter timeout;
- retry/expired-lease recovery where supported by the existing Task contract;
- duplicate/replay request;
- incompatible duplicate request;
- close/reopen recovery.

### 10.4 Repeated stability runs

Run at least:

- 20 repetitions of a dependency graph with `max_concurrency=1`;
- 20 repetitions with `max_concurrency>1`.

Across the repeated runs there must be zero unexpected:

- `STORAGE_CONFLICT`;
- `STORAGE_RECOVERY_REQUIRED`;
- task node `FAILED` caused by Task-state `STORAGE_CONFLICT`.

### 10.5 Recovery projection test

Construct or reproduce a state where all nodes are terminal while the durable graph/admission projection is still recoverable.

After reopen/recovery reconciliation, assert the graph is terminal and no longer returned by the recoverable-page index.

## 11. Documentation

Update public Task/Runtime documentation to state:

- SQLite-backed RuntimeState supports built-in durable TaskGraph execution without a SQLite-specific launcher or external lock;
- normal optimistic Task state-advancement conflicts are converged inside the Task domain;
- real ownership/fence/idempotency/integrity/storage errors remain observable;
- schema provisioning remains explicit and unchanged.

Do not document a new public API because none is added.

## 12. Versioning

Do not change `linktools-ai/linktools.yml` version.

Do not change dependency versions solely for this fix.

## 13. Validation gates

Run the targeted Task tests first, then repository gates:

```bash
python manage.py check linktools-ai
python manage.py build linktools-ai
python manage.py verify linktools-ai
```

After documentation changes, rerun the same gates.

## 14. Rollback

This change has no schema, durable-wire, or public-API migration.

Rollback procedure:

1. stop runtimes using the new code;
2. deploy the previous code revision;
3. reopen the existing `state.sqlite` unchanged.

No database rewrite is required.

Rollback intentionally reintroduces the previous TaskGraph concurrency defect.

## 15. Done criteria

The change is complete only when all are true:

- normal TaskGraph internal CAS races converge instead of leaking raw `STORAGE_CONFLICT`;
- same-owner/same-fence heartbeat cannot make a successful node execution persist as `FAILED(STORAGE_CONFLICT)`;
- cancellation/terminal races preserve the first valid durable terminal fact;
- `inspect_graph()` and `wait_graph()` are read-only;
- recovery still reconciles stale projections;
- SQLite single- and multi-concurrency public Runtime tests pass repeatedly;
- no pessimistic/global lock is introduced;
- no schema, wire, public API, dependency, or package-version change is introduced;
- repository validation gates pass.
