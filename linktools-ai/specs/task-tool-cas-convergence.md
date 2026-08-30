# TaskGraph / ToolOperation Optimistic CAS Convergence Spec

Status: Final revised

Baseline: `e1c8592230f5bcd48aee2c7db38064d8b0b7fca6`

This spec is subordinate to repository `AGENTS.md` and `linktools-ai/AGENTS.md`.

This change does **not** change the package version, persistence schema, durable wire format, or public Runtime API.

## 1. Problem

Runtime state uses optimistic `storage_version` CAS. SQLite makes concurrent writers serialize and therefore exposes semantic races reliably, but the defect is backend-neutral: normal Runtime actors can legitimately advance the same lease-backed record between another actor's read and CAS.

Two confirmed domains need convergence:

1. **TaskGraph**: scheduler, recovery, node claim/heartbeat/terminal, cancellation, and waiters interact with TaskGraph/TaskNode durable state.
2. **ToolOperation**: tool admission, lease heartbeat, terminal result/failure, and effect-unknown persistence interact with one ToolOperation durable record.

Database-native retryable aborts and semantic CAS conflicts remain separate:

```text
database retryable abort
    -> SqlStorageContext retry

semantic STORAGE_CONFLICT
    -> owning Runtime domain exits the failed transaction
    -> rereads latest durable state
    -> reclassifies the semantic state
    -> retries only the persistence mutation still permitted by that state
```

`STORAGE_CONFLICT` must not become a generic SQL retry condition.

## 2. Goals

The implementation must:

1. make SQLite-backed TaskGraph execution stable for `max_concurrency == 1` and `> 1`;
2. converge normal Task scheduler/recovery/heartbeat/terminal/cancellation CAS races internally;
3. make Task `inspect_graph()` and `wait_graph()` read-only observers;
4. preserve crash/reopen Task recovery and recovery-index repair;
5. converge ToolOperation admission/claim/heartbeat-terminal races without replaying tool side effects;
6. preserve Tool owner/fence/idempotency/result/effect-unknown semantic errors;
7. stop semantic Tool conflicts from being upgraded to `STORAGE_INTEGRITY_ERROR`;
8. keep transaction retry ownership at the method that owns the complete atomic invariant;
9. avoid pessimistic locks, hidden/global locks, new coordination records, schema changes, public API expansion, and version changes.

## 3. Non-goals

Do not add or change:

- `SELECT ... FOR UPDATE`, advisory locks, graph/tool mutexes, or process-global locks;
- WAL or `busy_timeout` as a semantic correctness fix;
- generic retry of `STORAGE_CONFLICT` in SQL storage;
- Task filesystem fallback;
- coordination tables/records;
- `Runtime.open(...)` launcher/repository injection;
- SQLite-specific Task/Tool APIs;
- database schema or durable wire format;
- package version or dependency versions solely for this fix;
- generic concurrency-framework refactors in unrelated Runtime domains.

Do not modify `storage/_database.py`, `storage/_dialects.py`, or `runtime/state/_sql.py` for this fix.

## 4. Core invariants

### 4.1 Transaction ownership

A transaction helper performs exactly one mutation attempt. It never catches a CAS miss and recursively re-enters `StateStore.mutate()` while the failed transaction is still active.

The method that owns the complete atomic invariant owns semantic retry:

```text
transaction A
  -> read
  -> compute
  -> CAS conflict
  -> transaction exits

transaction B
  -> fresh read
  -> recompute / reclassify
  -> retry only if latest state permits it
```

Never reuse records, `storage_version`, lease observations, or mutation candidates from the failed attempt.

### 4.2 Retry scope

Retry only raw `ErrorCode.STORAGE_CONFLICT` caused by optimistic record-version competition.

Do not retry semantic outcomes such as:

- `TASK_FENCE_STALE`;
- `TASK_OWNER_CONFLICT`;
- `TASK_NOT_READY`;
- `TOOL_OPERATION_CONFLICT`;
- `TOOL_RESULT_CONFLICT`;
- `TOOL_EFFECT_UNKNOWN`;
- `IDEMPOTENCY_CONFLICT`;
- `STORAGE_INTEGRITY_ERROR`.

### 4.3 Side-effect boundary

CAS convergence retries **durable persistence only**.

Never retry:

- `TaskNodeRunner.run()`;
- an already executed tool handler;
- shell/filesystem/network/MCP/external tool effects.

### 4.4 Cancellation

Caller cancellation may interrupt normal pre-effect work, but once an effect has produced an outcome and terminal durable persistence is in progress, Runtime must resolve durable truth before propagating cancellation.

## 5. TaskGraph state model

TaskNode durable records are authoritative for Task execution state.

`task_graph.status` and task-admission recovery state are derived persisted projections:

```text
TaskNode states
    -> task_graph.status
    -> task_admission recovery projection
```

Graph topology remains immutable.

## 6. TaskGraph production changes

Primary files:

- `runtime/state/_task_recovery_repository.py`
- `task/_service_impl.py`

`TaskRepositoryImpl` remains the base one-attempt semantic implementation. `DurableTaskRepositoryImpl` owns convergence for the materialized durable Task repository.

### 6.1 `reconcile_graph()`

`DurableTaskRepositoryImpl.reconcile_graph()` owns one atomic unit containing:

1. graph/node reread;
2. READY/BLOCKED derivation;
3. graph-status derivation;
4. node/graph CAS updates;
5. task-admission recovery projection synchronization.

On raw `STORAGE_CONFLICT`, the complete transaction exits and the entire unit restarts with a fresh read.

The private in-transaction helper performs one attempt only.

### 6.2 `cancel_graph()`

Cancellation uses the same outer fresh-transaction convergence.

On each attempt, terminal nodes are preserved and only currently non-terminal nodes are cancelled.

Required race semantics:

- completion wins first -> preserve that terminal node and cancel remaining eligible nodes;
- cancellation wins first -> old terminal writer fails fresh lease/fence validation;
- cancellation never rewrites an already terminal node.

### 6.3 `claim()`

A raw storage-version CAS miss starts a fresh claim attempt. The new attempt reclassifies the latest state through existing Task semantics.

Normal claim competition must not leak raw `STORAGE_CONFLICT`.

### 6.4 `renew()`

Keep existing fencing behavior unchanged.

Heartbeat CAS loss is `TASK_FENCE_STALE`. Do not automatically reacquire or blindly retry the lease.

### 6.5 `complete()` / `fail()`

A same-owner/same-fence heartbeat can advance `storage_version` between terminal read and terminal CAS.

On raw `STORAGE_CONFLICT`, retry only terminal persistence in a fresh transaction. Base live-lease validation determines whether the retry remains valid.

If owner, fence, status, or lease validity changed, preserve `TASK_FENCE_STALE`.

Never rerun the Task node.

### 6.6 Observer purity

`DefaultTaskService.inspect_graph()` and polling/readback in `wait_graph()` use `get_graph()` and do not reconcile.

If an authorized graph unexpectedly disappears between header authorization and graph read, report `STORAGE_INTEGRITY_ERROR`.

### 6.7 Recovery remains a writer

`recover_pending()` continues to call `reconcile_graph()` because crash/reopen recovery must repair stale graph/admission projections and remove terminal graphs from the recoverable index.

## 7. ToolOperation state model

ToolOperation lease ownership is represented by:

```text
status + owner + fence + lease_expires_at + storage_version
```

A heartbeat may legitimately advance only the lease expiry and storage version while keeping the same owner/fence.

Terminal identity additionally includes:

- terminal status;
- owner;
- fence;
- result payload, or error code/error payload.

A terminal row written by another owner/fence is not the current caller's successful commit even when the payload happens to match.

## 8. Tool repository convergence

Primary file:

- `runtime/state/_tool_repository.py`

`DurableToolRepositoryImpl` is materialized for `RuntimeDomain.RECOVERY` Tool operations.

Standalone methods may retry raw `STORAGE_CONFLICT` only after the base method's transaction has exited:

- `admit()`;
- `reserve()`;
- `claim()`;
- `complete_payload()`;
- `fail()`;
- `fail_payload()`;
- `mark_effect_unknown()`.

`renew()` is deliberately inherited unchanged: heartbeat CAS loss remains lease/fence loss semantics and is not automatically retried.

Each retry invokes the base method again so current lease expiry, status, owner, fence, aliases, and idempotency identity are reread and reclassified.

## 9. Tool outer command convergence

Primary file:

- `runtime/state/_tool_commands.py`

The exported `RuntimeStateCommands` retains the same Runtime-facing class name/API while specializing ToolOperation commands.

### 9.1 Why command-level ownership is required

Runtime's normal tool path executes terminal persistence through `RuntimeStateCommands.commit_tool_terminal()` using a storage-group transaction. Repository `*_in_transaction()` methods therefore cannot own retries without reusing the active transaction.

The command is the outer atomic owner and must retry only after the complete failed storage-group transaction exits.

### 9.2 `commit_tool_admission()`

On raw `STORAGE_CONFLICT`:

1. exit the failed transaction;
2. fresh-read the ToolOperation;
3. validate immutable admission identity;
4. return an already terminal compatible operation;
5. preserve `IDEMPOTENCY_CONFLICT`, `TOOL_EFFECT_UNKNOWN`, and `TOOL_OPERATION_CONFLICT`;
6. for `CLAIMED`/`PENDING`, re-enter a fresh repository/command transaction so lease expiry and ownership are classified inside the authoritative mutation path rather than guessed from readback.

### 9.3 `commit_tool_terminal()`

One attempt consists of one storage-group mutation invoking exactly one of:

- `complete_in_transaction()`;
- `fail_in_transaction()`.

If the attempt ends with raw `STORAGE_CONFLICT` and readback still observes `CLAIMED` with the same owner/fence, start a new storage-group transaction and retry terminal persistence only.

Readback classification:

| Durable observation | Outcome |
| --- | --- |
| expected terminal + same owner/fence + matching payload | committed |
| expected COMPLETED + same owner/fence + different result | `TOOL_RESULT_CONFLICT` |
| expected FAILED + same owner/fence + different error identity | `TOOL_OPERATION_CONFLICT` |
| terminal with different owner/fence | `TOOL_OPERATION_CONFLICT` |
| CLAIMED + same owner/fence after raw CAS miss | fresh terminal retry |
| CLAIMED + different owner/fence | `TOOL_OPERATION_CONFLICT` |
| EFFECT_UNKNOWN | `TOOL_EFFECT_UNKNOWN` |
| durable corruption | `STORAGE_INTEGRITY_ERROR` |
| genuinely unknowable commit | `STORAGE_COMMIT_UNKNOWN` |

Caller cancellation observed during an ambiguous terminal commit is remembered; Runtime resolves/commits terminal state first, then propagates `CancelledError`.

## 10. Tool bridge readback semantics

Primary file:

- `runtime/_tool.py`

`RuntimeToolOperationBridge._finish_with_readback()` must not classify valid semantic terminal conflicts as partial storage integrity failures.

Required behavior:

- same expected terminal, same owner/fence, matching payload -> committed;
- same expected terminal but different owner/fence -> `TOOL_OPERATION_CONFLICT`;
- same owner/fence but different completed result -> `TOOL_RESULT_CONFLICT`;
- same owner/fence but different failure identity -> `TOOL_OPERATION_CONFLICT`;
- EFFECT_UNKNOWN -> `TOOL_EFFECT_UNKNOWN`;
- only structurally impossible/corrupt persistence state becomes `STORAGE_INTEGRITY_ERROR`.

## 11. Runtime materialization

Primary file:

- `runtime/state/_materializer.py`

The materialized Recovery Tool repository must be `DurableToolRepositoryImpl`, just as Task uses its durable specialized repository.

No public repository injection is added.

## 12. Public behavior

No public method signatures are added or changed.

Public composition remains:

```text
RuntimeState / RuntimeStateRoute.sqlite(...)
    -> Runtime.open(...)
    -> built-in TaskGraph scheduler/recovery
    -> built-in ToolOperation durability
```

Downstream applications do not need private repositories, a SQLite-specific launcher, or external locking.

## 13. Error contract

Normal internal optimistic competition is converged and does not leak raw `STORAGE_CONFLICT` when the latest state still permits the operation.

Real semantic errors remain externally observable:

| Condition | Error |
| --- | --- |
| Task live owner conflict | existing Task owner conflict |
| Task stale/lost lease/fence | `TASK_FENCE_STALE` |
| Task not claimable | `TASK_NOT_READY` |
| Tool owner/fence/status conflict | `TOOL_OPERATION_CONFLICT` |
| Tool completed result mismatch | `TOOL_RESULT_CONFLICT` |
| Tool effect outcome unknown | `TOOL_EFFECT_UNKNOWN` |
| incompatible idempotent request | `IDEMPOTENCY_CONFLICT` / existing domain conflict |
| corrupt durable state | `STORAGE_INTEGRITY_ERROR` |
| unavailable storage | existing storage-availability error |
| genuinely unknowable commit | `STORAGE_COMMIT_UNKNOWN` |

## 14. I/O and concurrency constraints

The uncontended path must not add an extra transaction.

A semantic retry uses one fresh transaction per attempt.

Do not:

- sleep while holding a storage transaction;
- hold a coordination lock across Task or Tool business execution;
- add a background retry worker;
- add durable coordination records;
- add process-global synchronization.

`await asyncio.sleep(0)` is allowed only between failed transactions for event-loop fairness.

## 15. Required Task tests

Tests are mandatory.

Cover deterministic races for:

1. reconcile vs reconcile / node transition;
2. cancel vs terminal completion/failure;
3. claim vs competing state advancement;
4. heartbeat vs complete;
5. heartbeat vs fail;
6. stale owner/fence after terminal retry;
7. malformed/missing dependency integrity failure;
8. observer purity;
9. recovery-index repair after reopen.

Public SQLite Runtime coverage must include:

- single node with `max_concurrency=1`;
- dependency graph with `max_concurrency=1`;
- parallel graph with `max_concurrency>1`;
- dependency + parallel branch;
- failure/blocking;
- cancellation;
- timeout;
- duplicate/replay and incompatible duplicate;
- cold reopen/recovery.

Repeated stability acceptance:

- at least 20 dependency-graph runs with `max_concurrency=1`;
- at least 20 parallel runs with `max_concurrency>1`;
- zero unexpected `STORAGE_CONFLICT`, `STORAGE_RECOVERY_REQUIRED`, or Task failure caused by state CAS competition.

## 16. Required Tool tests

Cover deterministic behavior for:

1. concurrent compatible admission convergence;
2. admission CAS conflict -> fresh transaction/reclassification;
3. incompatible admission -> `IDEMPOTENCY_CONFLICT`;
4. heartbeat/version advancement vs complete -> terminal persistence succeeds;
5. heartbeat/version advancement vs fail -> terminal persistence succeeds;
6. heartbeat/version advancement vs `mark_effect_unknown()` -> effect-unknown persistence succeeds;
7. changed owner/fence -> `TOOL_OPERATION_CONFLICT` without retrying the old lease;
8. different completed result -> `TOOL_RESULT_CONFLICT`, never `STORAGE_INTEGRITY_ERROR`;
9. terminal owner/fence identity mismatch -> `TOOL_OPERATION_CONFLICT`;
10. caller cancellation during ambiguous terminal persistence -> durable terminal state is resolved before `CancelledError` propagates;
11. persistence retry does not invoke the tool handler/effect a second time.

## 17. Documentation

Public Runtime documentation must state:

- SQLite-backed RuntimeState supports durable built-in TaskGraph execution without SQLite-specific coordination;
- normal internal Task CAS races converge inside Runtime;
- ToolOperation terminal persistence is lease-aware;
- same-lease heartbeat/terminal races do not replay the tool effect;
- real ownership/fence/idempotency/result/effect-unknown/integrity/storage errors remain observable.

No new public API is documented because none is added.

## 18. Versioning

Do not change `linktools-ai/linktools.yml` version.

Do not change dependency versions solely for this fix.

## 19. Validation gates

Run targeted Task and Tool tests, then repository gates:

```bash
python manage.py check linktools-ai
python manage.py build linktools-ai
python manage.py verify linktools-ai
```

After documentation changes, rerun the same gates.

## 20. Rollback

This change has no schema, durable-wire, or public-API migration.

Rollback:

1. stop runtimes using the new code;
2. deploy the prior code revision;
3. reopen the same durable Runtime state unchanged.

No database rewrite is required. Rollback intentionally reintroduces the previous TaskGraph/ToolOperation concurrency defects.

## 21. Done criteria

The change is complete only when all are true:

- Task internal CAS races converge without leaking normal raw `STORAGE_CONFLICT`;
- Task heartbeat/terminal and cancel/terminal races preserve fencing and terminal truth;
- Task observation paths are read-only;
- Task reopen recovery repairs stale projections;
- Tool admission/claim/terminal/effect-unknown CAS races converge through fresh transactions;
- Tool heartbeat remains fencing-only and never reacquires ownership automatically;
- persistence retry never replays Task node or Tool business effects;
- Tool result/owner/fence conflicts retain semantic errors rather than becoming storage-integrity errors;
- SQLite Task stability/recovery tests pass;
- deterministic Tool convergence tests pass;
- no pessimistic/global lock is introduced;
- no schema, wire, public API, dependency, or package-version change is introduced;
- repository validation gates pass.
