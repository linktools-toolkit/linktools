# TaskGraph / ToolOperation Optimistic Concurrency Convergence Spec

Status: Final revised

Baseline: `e1c8592230f5bcd48aee2c7db38064d8b0b7fca6`

This spec is subordinate to repository `AGENTS.md` and `linktools-ai/AGENTS.md`. It does not change the package version, persistence format, database schema, or public Runtime API.

## 1. Problem

When `RuntimeDomain.TASK` uses a durable `StateStore`, built-in Runtime actors can concurrently update the same TaskGraph or TaskNode records. SQLite makes this easy to reproduce because writers are serialized, but the correctness defect is backend-neutral: Runtime semantic CAS conflicts are not consistently converged by the domain that owns the invariant.

The same defect class also exists for durable ToolOperation records: a lease heartbeat can advance `storage_version` while the same owner/fence is committing `COMPLETED`, `FAILED`, or `EFFECT_UNKNOWN`.

Database transaction conflicts and semantic CAS conflicts remain separate concerns:

```text
database retryable abort
    -> SqlStorageContext retry

semantic STORAGE_CONFLICT
    -> owning Runtime domain rereads durable state
    -> recomputes semantic decision
    -> starts a fresh transaction if retry is valid
```

`STORAGE_CONFLICT` must not become a generic SQL retry condition.

## 2. Goals

The implementation must:

1. make SQLite-backed TaskGraph execution stable for `max_concurrency == 1` and `> 1`;
2. prevent normal scheduler/recovery/heartbeat/terminal CAS races from leaking `STORAGE_CONFLICT`;
3. preserve crash/reopen TaskGraph recovery;
4. preserve real ownership, fencing, idempotency, integrity, and storage errors;
5. converge ToolOperation heartbeat races with complete/fail/effect-unknown without replaying the tool effect;
6. keep observers read-only;
7. avoid pessimistic locks, hidden global locks, new coordination records, schema changes, and public API expansion.

## 3. Non-goals

Do not add or change:

- `SELECT ... FOR UPDATE`, advisory locks, graph-level/global mutexes, or SQLite-specific locks;
- WAL or `busy_timeout` as a semantic correctness fix;
- generic retry of `STORAGE_CONFLICT` in SQL storage;
- Task filesystem fallback;
- Task coordination tables/records;
- `Runtime.open(...)` launcher injection or SQLite-specific Task launcher/configuration;
- database schema or durable wire format;
- package version;
- unrelated Runtime domain concurrency abstractions.

Do not modify `storage/_database.py`, `storage/_dialects.py`, or `runtime/state/_sql.py` for this change.

## 4. Core invariants

### 4.1 Transaction ownership

A transaction helper performs exactly one mutation attempt. It never retries by recursively entering `StateStore.mutate()`.

The method that owns the complete atomic invariant owns semantic retry. A retry must occur only after the previous transaction exits and must reread all durable state used to make the decision.

```text
transaction A
  -> read
  -> compute
  -> CAS conflict
  -> rollback/exit
transaction B
  -> fresh read
  -> recompute
  -> retry only if the latest semantic state permits it
```

Never reuse stale records, `storage_version`, or mutation candidates across attempts.

### 4.2 Task authority

TaskNode records are authoritative for Task execution state. `task_graph.status` and durable Task admission recovery state are derived projections/indexes.

```text
TaskNode states
    -> task_graph.status
    -> task_admission recovery projection
```

Graph topology remains immutable.

### 4.3 Lease/fence semantics

Heartbeat failure never reacquires ownership automatically. Terminal retry is allowed only when a fresh read proves the same live owner/fence still owns the operation and the conflict was only a storage-version advancement.

## 5. Task repository refactor

Production files:

- `linktools-ai/src/linktools/ai/runtime/state/_repositories.py`
- `linktools-ai/src/linktools/ai/runtime/state/_task_recovery_repository.py`
- `linktools-ai/src/linktools/ai/task/_service_impl.py`

`linktools-ai/src/linktools/ai/task/_local.py` should not require semantic conflict swallowing. If expected repository CAS reaches the launcher, repository convergence is incomplete.

### 5.1 Transaction helpers

Move current one-attempt mutation bodies into private transaction helpers:

```python
async def _reconcile_graph_in_transaction(
    self,
    transaction: StateTransaction,
    graph_id: str,
    *,
    tenant_id: str,
) -> TaskGraphView: ...

async def _cancel_graph_in_transaction(
    self,
    transaction: StateTransaction,
    graph_id: str,
    *,
    tenant_id: str,
) -> TaskGraphView: ...
```

These helpers must not retry or open nested transactions.

Apply the same one-attempt/helper pattern where needed for `claim()` and Task terminal persistence so the public method can start a fresh transaction after a raw CAS miss.

### 5.2 DurableTaskRepository owns aggregate retry

`DurableTaskRepositoryImpl` is the actual materialized Task repository. Its invariant includes both TaskGraph mutation and recovery projection synchronization.

`reconcile_graph()` must retry the complete unit:

```text
fresh transaction
  -> _reconcile_graph_in_transaction
  -> _sync_recovery_projection
```

On `STORAGE_CONFLICT`, the transaction exits and the whole unit is retried from a fresh read.

`cancel_graph()` follows the same rule.

Do not retry inside the base helper while a durable outer transaction is active.

### 5.3 Reconcile convergence

A retry rereads the graph and all nodes, then recomputes READY/BLOCKED and aggregate graph status from the latest state. Normal competing progress must converge rather than leak `STORAGE_CONFLICT`.

### 5.4 Cancel convergence

If completion commits first, cancellation rereads and preserves the terminal node while cancelling remaining non-terminal nodes.

If cancellation commits first, an old terminal writer must fail lease validation and surface `TASK_FENCE_STALE`; it must never overwrite CANCELLED.

### 5.5 Claim convergence

A raw CAS miss in `claim()` starts a fresh transaction and reclassifies latest state. Final outcomes are semantic (`TaskLease`, `TASK_OWNER_CONFLICT`, `TASK_NOT_READY`, fencing/integrity/storage errors), not raw `STORAGE_CONFLICT` for normal competition.

### 5.6 Renew remains fencing-only

`renew()` keeps current semantics. A CAS miss remains `TASK_FENCE_STALE`; it must not retry or reacquire the lease.

### 5.7 Complete/fail heartbeat race

After a terminal CAS miss, start a fresh transaction and reread the node. If it is still `RUNNING`, has the same owner/fence, and the lease is live, retry only terminal persistence using the latest storage version.

If status, owner, fence, or lease validity changed, return `TASK_FENCE_STALE`.

Never replay `TaskNodeRunner.run()`.

### 5.8 Retry policy

Do not use a fixed `MAX_CAS_RETRIES`. Each retry must be a fresh semantic observation. `await asyncio.sleep(0)` is allowed only for event-loop fairness; do not add fixed/exponential sleeps inside the semantic convergence loop.

## 6. Task service observer/recovery behavior

### 6.1 Read-only observers

`DefaultTaskService.inspect_graph()` must use `get_graph()`, not `reconcile_graph()`.

`DefaultTaskService.wait_graph()` polling and scheduler-failure readback must use `get_graph()`, not `reconcile_graph()`.

A waiter or inspector must never advance Task state or recovery projections.

### 6.2 Recovery remains a writer

`recover_pending()` must continue to call `reconcile_graph()`.

A runtime can crash after TaskNode terminal state is durable but before graph/admission projection is refreshed. Reopen recovery must recompute graph status and repair the recovery index so terminal graphs disappear from recoverable enumeration.

## 7. ToolOperation convergence

Production files:

- `linktools-ai/src/linktools/ai/runtime/state/_repositories.py`
- `linktools-ai/src/linktools/ai/runtime/state/_commands.py`
- `linktools-ai/src/linktools/ai/runtime/_tool.py`

Classify Tool writers as follows:

| Mutation | Conflict behavior |
| --- | --- |
| admission | fresh read; identical replay succeeds; incompatible identity conflicts |
| claim | fresh read and semantic reclassification |
| renew | no semantic retry; ownership/fence conflict |
| complete | same-live-lease heartbeat conflict may retry terminal persistence |
| fail | same-live-lease heartbeat conflict may retry terminal persistence |
| effect unknown | same-live-lease heartbeat conflict may retry terminal persistence |

### 7.1 Admission

Concurrent identical admission must converge to the same durable ToolOperation. A CAS/alias race must fresh-read and return the existing record when identity matches.

Incompatible admission remains `IDEMPOTENCY_CONFLICT`.

Both `ToolRepositoryImpl.admit()` and `RuntimeStateCommands.commit_tool_admission()` must expose the same semantics.

### 7.2 Claim

A raw CAS miss starts a fresh transaction and re-evaluates the latest operation. Final outcomes are semantic (`ToolOperationRecord`, `TOOL_OPERATION_CONFLICT`, `TOOL_EFFECT_UNKNOWN`, `IDEMPOTENCY_CONFLICT`, integrity/storage errors), not raw `STORAGE_CONFLICT` for normal competition.

### 7.3 Renew

Keep current fencing behavior. Do not auto-reacquire after a heartbeat conflict.

### 7.4 Complete/fail/effect-unknown heartbeat race

If heartbeat advances the record version while the same owner/fence is terminalizing, a fresh read may prove:

```text
status == CLAIMED
owner == expected owner
fence == expected fence
lease still live
```

Only then may terminal persistence be retried with a new transaction/latest version.

If ownership/fence/status/lease changed, surface the existing Tool ownership/conflict semantics.

The actual tool handler/effect must never be replayed.

`mark_effect_unknown()` follows the same rule. Repeated identical `EFFECT_UNKNOWN` from the same owner/fence/error is idempotent success.

### 7.5 Exact terminal identity

A terminal readback counts as the caller's committed result only when status, owner, fence, and result/error identity all match.

A terminal record written by a different owner/fence is not success even if payload bytes happen to match.

### 7.6 Preserve semantic errors

Valid competing terminal outcomes are not storage corruption.

- same owner/fence, different completed payload -> `TOOL_RESULT_CONFLICT`;
- same owner/fence, incompatible failed outcome -> `TOOL_OPERATION_CONFLICT`;
- different owner/fence -> `TOOL_OPERATION_CONFLICT`;
- durable `EFFECT_UNKNOWN` -> `TOOL_EFFECT_UNKNOWN`.

Only impossible partial state or violated durable invariants map to `STORAGE_INTEGRITY_ERROR`.

Apply this consistently to both `RuntimeStateCommands.commit_tool_terminal()` and `RuntimeToolOperationBridge._finish_with_readback()`.

### 7.7 Cancellation

Caller cancellation does not decide durable truth. If cancellation arrives after the external/tool effect has executed, terminal convergence must first make `COMPLETED`, `FAILED`, or `EFFECT_UNKNOWN` durably observable (or establish ownership loss/unknown commit) before cancellation is propagated.

No CAS retry may re-enter the tool handler.

## 8. Unchanged Runtime domains

Do not refactor Execution, Session, Recovery/Approval, Step, or Transcript concurrency as part of this change. Their current conflict handling may be tested for regression but is outside production scope unless implementation reveals a direct dependency required by the invariants above.

## 9. Error contract

| Condition | Required behavior |
| --- | --- |
| normal internal semantic CAS race | converge internally |
| Task lost lease/fence | `TASK_FENCE_STALE` / existing Task ownership error |
| Task not runnable | `TASK_NOT_READY` |
| Tool lost ownership/fence | `TOOL_OPERATION_CONFLICT` |
| incompatible Tool result | `TOOL_RESULT_CONFLICT` |
| Tool effect unknown | `TOOL_EFFECT_UNKNOWN` |
| incompatible idempotent request | `IDEMPOTENCY_CONFLICT` |
| durable corruption | `STORAGE_INTEGRITY_ERROR` |
| infrastructure unavailable | existing storage availability error |
| genuinely unknowable commit | `STORAGE_COMMIT_UNKNOWN` |

Normal heartbeat/version competition must not leak `STORAGE_CONFLICT`.

## 10. Required tests

Tests are mandatory acceptance criteria for this change.

### 10.1 Public Runtime SQLite TaskGraph

Use public APIs (`RuntimeStateRoute.sqlite`, `Runtime.open`, `Runtime.run_graph`, `Runtime.run_graph_and_wait`) and cover:

- single node, `max_concurrency=1`;
- dependency `A -> B`, `max_concurrency=1`;
- parallel graph, `max_concurrency>1`;
- mixed dependency/parallel graph;
- success, failure, cancellation, timeout, reclaim/retry, duplicate/replay, incompatible duplicate, close/reopen, crash recovery.

Run the dependency graph at least 20 consecutive times and the parallel graph at least 20 consecutive times. Require zero leaked `STORAGE_CONFLICT`, unexpected `STORAGE_RECOVERY_REQUIRED`, Task failures caused by Task-state storage conflicts, orphan RUNNING nodes, or stale terminal recovery entries.

### 10.2 Deterministic Task CAS tests

Use barriers/fault injection to force:

- reconcile vs reconcile;
- reconcile vs claim;
- reconcile vs terminal;
- cancel vs complete/fail;
- claim vs claim;
- heartbeat vs complete/fail.

Assert final durable state, not merely retry count.

Heartbeat/terminal tests must prove `TaskNodeRunner.run()` executes exactly once.

A fencing test must force a new owner/fence after the first terminal conflict and require the old writer to receive `TASK_FENCE_STALE`.

### 10.3 Recovery projection

Force node terminal durability with stale graph/admission projection, reopen Runtime, run recovery, and assert the graph becomes terminal and disappears from recoverable enumeration.

### 10.4 Observer purity

With no scheduler progression, `inspect_graph()` and `wait_graph()` polling must not change TaskNode, TaskGraph, or Task admission `storage_version`, owner, lease, or fence.

### 10.5 Deterministic Tool tests

Force:

- heartbeat vs complete;
- heartbeat vs fail;
- heartbeat vs effect-unknown;
- owner/fence loss before terminal retry;
- conflicting terminal result;
- concurrent identical admission;
- incompatible admission.

Require semantic outcomes described above and no erroneous `STORAGE_INTEGRITY_ERROR` for valid conflicts.

### 10.6 Tool side-effect exactly-once

Use a tool handler that increments an observable counter, force heartbeat/terminal CAS competition, and require the counter to remain exactly `1` regardless of terminal persistence retries.

### 10.7 Tool cancellation

Cancel the caller after the handler has produced its result while terminal persistence is competing. Require durable terminal ToolOperation state before cancellation becomes visible, with no handler replay and no stranded `CLAIMED` record.

## 11. I/O and performance constraints

Uncontended paths must not gain unnecessary transactions.

Task observers become read-only, reducing writes.

Each semantic retry is one fresh mutation transaction. Do not sleep while holding a transaction, execute Task business logic in a transaction, execute Tool side effects in a transaction, or add a background retry worker.

## 12. Public API and persistence compatibility

Keep unchanged:

- `Runtime.open(...)`;
- `Runtime.run_graph(...)` / `run_graph_and_wait(...)`;
- `Runtime.task.*`;
- `RuntimeState` / `RuntimeStateRoute.sqlite(...)`;
- TaskGraph/TaskGraphLimits/TaskGraphLauncher public signatures;
- existing Tool public Runtime behavior;
- database schema and persisted wire format;
- package version.

The implementation must remain backend-neutral across SQLite, MySQL, PostgreSQL, filesystem, and memory StateStore implementations that satisfy the existing CAS contract.

## 13. Documentation

Update public documentation to state that SQLite-backed RuntimeState supports durable TaskGraph execution without a SQLite-specific launcher or external lock, and that normal built-in optimistic state-advancement races are resolved internally while real ownership/fencing/idempotency/integrity/storage failures remain observable.

Document that durable ToolOperation terminal persistence is lease-aware and that heartbeat/terminal reconciliation never replays the Tool effect.

## 14. Validation gates

Run from repository root:

```bash
python manage.py check linktools-ai
python manage.py build linktools-ai
python manage.py verify linktools-ai
```

Run all new deterministic concurrency/recovery tests and existing affected Runtime/Task/Tool tests.

## 15. Review checklist

Review every Task writer: reconcile, cancel, claim, renew, complete, fail, recovery projection, wait, inspect, recover_pending, scheduler.

Review every Tool writer: admit, claim, renew, complete, fail, effect-unknown, terminal command, bridge terminal readback.

For each, verify:

- retry ownership is at the complete atomic invariant boundary;
- transaction helpers do not retry recursively;
- every retry uses a fresh durable read;
- fencing cannot be bypassed;
- Task business logic and Tool side effects are never replayed;
- semantic conflicts are not converted into storage integrity failures;
- no pessimistic/global lock or schema/API/version change was introduced.

## 16. Rollback

There is no schema, durable format, public API, or version change. Rollback consists only of stopping the fixed Runtime and deploying the previous implementation against the same durable state. The previous concurrency defect will then reappear by design.

## 17. Definition of Done

The change is complete only when:

1. SQLite TaskGraph is stable at concurrency 1 and >1;
2. expected internal Task CAS races do not leak `STORAGE_CONFLICT`;
3. observers are read-only and recovery projection repair remains correct;
4. Task heartbeat/terminal and fencing tests pass;
5. Tool heartbeat/complete, heartbeat/fail, and heartbeat/effect-unknown converge;
6. Tool side effects are never replayed by persistence retry;
7. identical Tool admission converges and incompatible admission remains a semantic conflict;
8. real Tool result/ownership conflicts keep the correct Tool error and are not misclassified as storage corruption;
9. crash/reopen and duplicate/replay tests pass;
10. no pessimistic/global lock, schema change, public API expansion, or package version change exists;
11. `manage.py check`, `build`, and `verify` pass;
12. public behavior documentation is updated.
