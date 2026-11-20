# Domain composition and ownership

The executable [domain composition example](../examples/domain_composition.py)
uses current public APIs and an offline SDK model. It does not propose a new
engine, registry, adapter, or application lifecycle. Every model binding, task,
score function, graph input, shared resource, result read, and close is visible.
The temporary SQLite stores are for the example, not a production migration.

For real capability tool calls, captured Asset resources, native wrappers,
Memory reuse after reopen and Session/Planning composition, see the complete
[capability program and ownership matrix](capability-ownership.md).

Run it after installing the package through the repository entry point:

```sh
python manage.py install --editable
python linktools-ai/examples/domain_composition.py
```

## Current and retained application paths

| Scenario | Current public entry points | Retained call shape | Owner and close |
| --- | --- | --- | --- |
| Simple Agent | `RuntimeFeatures.model`, `RuntimeFeatures.agent`, `Runtime.open`, `runtime.agents.get().start`, `Execution.wait` | No extra host adapter or manager | Runtime owns its Agent producer and its opened Persistence |
| Ordinary Task | `Task`, `TaskGraph`, `TaskNode`, `TaskEngine.open`, `engine.start`, `graph.wait`, `graph.result` | No Agent or Runtime required | TaskEngine owns storage it opens; a supplied Persistence remains host-owned |
| Mixed Task graph | `Task.from_agent`, `AgentTaskInput`, `TaskNodeResultRef` | Same Agent handle and same Persistence; explicit result input refs | Task owns scheduling; Agent remains the execution producer; closing Task does not close Agent |
| Evaluation of a Task | `Evaluations.open`, `publish_dataset`, `start`, `run.wait`, `create_report` | Same ordinary Task producer; no Agent dependency | Evaluations owns the engine it opens, or borrows the supplied engine |
| Evaluation of an Agent | `Task.from_agent` plus the same Evaluation entry points | Agent is a target adapter, not a second evaluator | Close evaluation coordination before Task, then Agent |
| Metrics alone | `Metrics.in_memory`, `define`, `record`, `query` | No execution engine or automatic recorder required | Metrics borrows its store and has no producer close; externally owned SQL engines stay host-owned |
| Metrics in Agent/Task | Existing `metrics=provider` argument | Pass the same provider to both domains | Each producer closes only its own recorder, not its peer's recorder or the shared provider |
| One-context composition | `Runtime.open`, `runtime.tasks.bind`, `runtime.evaluations.bind` | One host `async with`; no extra mandatory contexts | Runtime closes composed Evaluation, Task, Agent, then owned Persistence |
| Shared independent domains | `Persistence.open`, `Runtime.open(persistence=...)`, `TaskEngine.open` | Explicit sharing is optional advanced composition | Attached Runtime and TaskEngine borrow the already-open Persistence |

The program exercises each composition and ownership boundary through the
public APIs. TaskEngine uses the existing `AgentExecutionPort`, exposed by
`Agent.execution_port` and `runtime.agent_execution`, for Agent-backed nodes.
That port preserves the Agent producer's execution authority.

## Important signatures and semantics

- `Runtime.open(namespace, *, storage, tenant_id="default", context=None,
  features=None, metrics=None, authorization=None, auto_recover=False)` returns
  the context-managed Runtime. Omitting features does not start an Agent
  producer. Pass `auto_recover=True` to opt into startup-wide execution
  reconciliation. `Runtime.open(persistence=..., ...)` borrows an already-open,
  writable Persistence and inherits its namespace and tenant. Do not pass a
  namespace, tenant, or storage configuration in borrowing mode. The legacy
  `Runtime.attach(persistence=...)` forwards to this same entry. Retain the
  returned object when retrying cleanup after partial initialization failure.
- `RuntimeFeatures.model(name, binding)` accepts a public `ModelBinding` or a
  synchronous loader. A binding describes logical model behavior and
  materializes the actual SDK model. The offline binding in the example is
  complete, not a hidden test helper or a suggested production model provider.
- `Task.from_agent(id, agent, ...)` refers to an existing Agent producer.
  TaskEngine requires its exact Persistence owner, not merely an equivalent
  database URL. A graph's original principal remains the resumed execution
  principal; a recovery actor authorizes recovery, not a new execution identity.
- `TaskNode(..., dependencies=(...))` orders execution only. To consume a
  predecessor's output, declare `input_refs={name: TaskNodeResultRef(node_id)}`
  and call `context.read_dependency(name)`. Ordering alone does not concatenate
  all earlier results into the model prompt.
- `engine.start(graph, *, principal=..., idempotency_key=...)` returns a graph
  handle. `await graph.wait(timeout_seconds=...)` returns a `WaitResult` whose
  `result` is the graph view; `await graph.result(node_id)` reads a node output.
  `Execution.wait` also returns a `WaitResult`, not a raw `ExecutionResult`.
  Replacing service result reads with this stream-aware wait is not equivalent.
- `Evaluations.open(engine, *, assets=())` owns evaluation coordination and
  borrows its engine. `runtime.evaluations.bind(engine, *, assets=())` instead
  registers that coordinator with Runtime's existing close sequence. The
  example supplies an explicit service principal for evaluation control; it
  does not weaken the authorization policy.
- Session and Execution management authorize the record read for that external
  operation. The default policy checks both tenant and resource owner; storage
  tenant scoping alone does not isolate callers within one tenant. Hosts can
  supply a custom policy, including Evaluation actions. Policy decisions are
  never cached across operations. Mutations retain revision and immutable-owner
  checks after awaited hooks. Tool filesystem and shell permissions remain a
  separate execution and sandbox boundary.
- `Metrics.record(metric, value, *, observation_id=..., ...)` settles an
  explicit source write. Automatic producer recording is a separate recorder
  lifetime. A host can supply a different `MetricRecorderProvider`; a dropped
  metric must not become a different execution outcome.
- Agent recovery uses the official
  `pydantic_ai_harness.step_persistence.StepPersistence(store=steps,
  capture_frontier=True)` capability. Register it at Runtime scope alongside
  `TraceCapability()` when both native checkpoints and execution evidence are
  needed. Its StepStore is caller-owned and must outlive Runtime. The example's
  `SqliteStepStore(database=path)` opens and closes short-lived connections per
  operation; it has no store-level close/context API. A caller-supplied
  `connection=...` must instead be closed by that caller after Runtime closes.
  Native snapshots are not copied into a second Runtime checkpoint archive.

## Resource ownership is selected per resource

An application that needs only Task or Evaluation does not have to open
Runtime or manage a separate Persistence context:

```python
async with TaskEngine.open(task, namespace="jobs", storage=storage) as engine:
    run = await engine.start(graph, idempotency_key="start")
    await run.wait()

async with Evaluations.open(
    namespace="evaluations", storage=storage, definitions=(target, scorer),
) as evaluations:
    run = await evaluations.start(request)
    await run.wait()
```

When sharing resources, pass an open `Persistence` to `TaskEngine.open`, or
pass an existing engine to `Evaluations.open(engine)`. Each borrower holds a
reference until its cleanup completes. A close request with live borrowers
fails without waiting or closing the dependency; close the borrowers and
retry. Finishing the last borrow does not close anything on behalf of the
owner. Owned resources close in dependency order: Evaluation, TaskEngine,
then Persistence. For an Agent-backed task, use `Task.from_agent` with an
explicit Agent and open the TaskEngine over that Agent's Persistence; neither
independent entry point starts an Agent Runtime implicitly.

`Runtime.open(persistence=...)` borrows Persistence, not every object in `features`. Resource
declarations still determine who initializes and closes each dependency:

| Resource supplied to Runtime | Initialization and close owner | Existing public call shape |
| --- | --- | --- |
| Persistence opened by `Runtime.open` | Runtime | One Runtime context |
| Persistence supplied to `Runtime.open` | Host; it must already be open and writable | Outer Persistence context, inner Runtime context |
| `AssetCapability(store)` in `RuntimeFeatures` | Runtime, even with `attach` or an already initialized store | Pass the declaration directly for ownership transfer |
| Asset source in `features.borrow()` | Host; initialize before borrowers and close after all borrowers finish | Pass the live borrowed source, not an inspection resolution |
| Default `StepPersistenceCapability()` adapter | Runtime creates/closes the adapter over Execution storage; its external SQL engine remains host-owned | No extra store context; requires an Execution SQL route |
| Explicit StepStore in `StepPersistenceCapability(store=...)` or native `StepPersistence(store=...)` | Caller; Runtime does not close it | Keep the store usable through producer cleanup |

Runtime releases owned dependencies only after producer cleanup succeeds. A
producer close failure leaves those dependencies available for retrying
`runtime.close()`; closing borrowed Persistence first would remove that recovery
path. Partial initialization also uses the same cleanup ownership rather than
transferring it to the failing capability. The existing
`test_runtime_resource_ownership.py` checks these combinations, explicit stores
inside wrappers, partial initialization and close retry; this table does not add
a new lifecycle or automatic recovery mechanism.

## Recovery uses the existing owner

These cases have different identities and must not be combined into a generic
"restore Memory" path:

| Operation | State and resource owner | Existing contract coverage |
| --- | --- | --- |
| Retry an ordinary execution after Runtime reopens | Execution retains its admitted Memory scope; current capability declarations determine available tools. A removed Memory declaration stays absent, and a same-named business tool does not become a Memory tool. | `test_memory_capability_scope.py::test_memory_tools_retry_resolves_current_tools_but_retains_admitted_scope` |
| Retry an imported execution | The new execution gets an isolated scope seeded with the captured content and versions, not later edits in the original execution's scope. | `test_imported_context_retry.py::test_retry_imported_memory_uses_fresh_isolated_scope` |
| Continue a deferred imported execution | The same execution keeps its scope and intervening mutations; continuation does not reapply the seed over those mutations. | `test_memory_capture_replay.py::test_runtime_imported_memory_resumes_deferred_without_rollback` |
| Reopen and retry a Memory seed | `PersistentMemoryStore` uses the original seed receipts; Persistence owns their atomic commit with the seeded paths. | `test_memory_capture_replay.py::test_memory_capture_receipt_replays_after_sqlite_reopen` and `test_memory_capture_retry_resolves_committed_partial_seed` |
| Replace a custom native persistent capability during a run | The SDK selects the effective run-local instance. Returning the declaration template or an instance already bound to another run is rejected before changing the persistence callback. New submissions use the effective instance; already accepted writes retain their original callback owner until commit. | `test_capability_persistence_lifecycle.py::test_native_persistence_replacements_keep_current_and_accepted_write_owners` and `test_native_replacement_cannot_reuse_another_persistence_owner` |

The deferred-continuation case covers one Runtime lifetime. The seed-receipt
case separately covers SQLite reopen; together they do not prove every
cold-process continuation scenario. Accepted owner-local Python callbacks are
not a cross-worker durable callback queue. Extending recovery requires a real
caller contract, not another registry or a second copy of Memory state.

## Failure, cancellation, and recovery boundaries

A wait timeout or a cancelled observer is not proof that the producer stopped.
Use the graph/execution's explicit `cancel` operation when producer cancellation
is intended; Task's execution handle keeps its Task cancellation settlement.
Preserve the existing graph and execution idempotency keys when resolving an
uncertain outcome instead of blindly starting a new execution.

The Task-facing Agent execution port exposes
`request_terminal_handoff(execution_id, *, tenant_id, terminal_commit=False)`.
The keyword matches the concrete producer operation and the Task read adapter;
it forwards the existing terminal-commit flag to producer-owned handoff
cleanup, without bypassing outstanding dependency holds.

A Task reference retains execution data until its durable binding/handoff is
established. Closing a borrower must neither erase this obligation nor close a
peer producer. Recovery still uses the existing durable graph/binding and
original principal. Internal layering changes must not replace these semantics
with a separate state authority, copied result, or new recovery engine.

The examples show ordinary successful completion and real close behavior.
They are not a fault-injection suite, a production readiness claim, a benchmark,
or evidence that every capability's binding and restore owner has been migrated.
Capability-specific recovery, partial initialization and cancellation regression
coverage remains in the repository tests.

## Agent-node adaptation

TaskEngine composes graph services and binds declarations. AgentTaskRunner owns
Agent-node output selection, input capture, producer start/get, invocation
recording, dependency-hold acquisition and Agent-specific recovery adaptation.
It receives the original producer
port and the Task services already selected by the engine, rather than a second
set of independent callbacks for the same operations.

The runner captures the compiled declaration and binding once. It reuses the
engine's admission and budget repositories and records the original port for
later reads and recovery. The node executor is resolved lazily because it is
constructed from the bound Task definitions. This does not add another public
adapter, producer, resource owner or user-managed lifecycle.

The port exposes producer read/control, retention and history operations, plus
`get(..., cancel=...)` for Task's graph-aware cancellation callback. Its service
and execution-binding collaborators remain private and may differ internally.
Task does not impose an equality check or acquire another producer lifecycle.

Recovery retains three phases: resolve all admitted Agent bindings, validate
all referenced executions, then dispatch eligible producers. The Task executor
owns graph authorization, admission, original-principal checks, node fencing
and post-dispatch readback. AgentTaskRunner projects each node's output contract
onto its admitted Agent binding and uses the original port for resolution and
takeover. It also retains the live invocation-capture handoff exception and
the original execution deadline; an expired execution is reattached for
settlement without dispatching another model or tool.

Reattachment uses the graph's admitted principal. The recovery operation's
request identity instead records the authorized recovery actor. No new
application call, port method, persisted contract or lifecycle is required.

## Retained ownership and verification

- MCP protocol/client ownership is separate from executable tool Boundary,
  effect facts and recorded response fixtures. Fixture playback creates no live
  MCP client. Stable model naming stays with the execution view, resources
  with Asset/Workspace; do not move fixture authority into the protocol client.
- Task's Agent execution port now covers admission/capture, observation/history,
  retention and cancellation through public operations. Preserve these complete
  operation and lifetime contracts when changing domain boundaries.
- `memory.PersistentMemoryStore` owns persistent Memory operations, mutation
  receipts and capture/seed. Runtime selects the admitted scope and resources,
  then delegates capture and restore to that adapter. Persistence retains
  transaction authority, and the adapter borrows its resources. Planning owns
  its store and native projection; Runtime selects the run owner. Model plan
  tools, execution plan mode and Task scheduling remain three different concepts.
- The Step capability owns native SDK adaptation and attempt binding, including
  preserving custom SDK hooks. The execution adapter chooses the attempt
  metadata, transaction boundary and store. This does not add a checkpoint
  writer or change the recovery authority.
- A consumed, failed Evaluation trial-context exit is still reported. When
  all actual borrowers and observations have drained, it no longer prevents
  closing independent readers, TaskEngine and owned Persistence. A close
  that still has pending work or has failed a retryable cleanup retains the
  resources needed for that cleanup.

The runnable examples and regression tests cover different parts of the contract:

| Boundary | Runnable example or focused regression (under `tests/ai`) |
| --- | --- |
| Independent and composed use, same storage and ownership | [domain_composition.py](../examples/domain_composition.py), `test_runtime_open_persistence.py`, `test_domain_composition_ownership.py` |
| Scope exit errors, partial cleanup retry and close order | `test_evaluation_trial_scopes.py`, `test_evaluation_lifecycle.py`, `test_runtime_owner_release.py` |
| Capability/Agent declarations, dynamic discovery and retained resources | [capability_composition.py](../examples/capability_composition.py), `test_current_runtime_features.py`, `test_runtime_capability_scope.py` |
| Memory/Planning, MCP and Subagent recovery boundaries | `test_memory_capture_replay.py`, `test_planning_store_binding.py`, `test_mcp_lifecycle.py`, `test_subagent_parent_recovery.py` |
| Checkpoint, custom effective run instance and atomic commits | `test_step_attempt_binding.py`, `test_official_runtime_recovery.py`, `test_capability_persistence_lifecycle.py`, `test_capability_atomic_writes.py` |
| Task-to-Agent port, handoff and independent Metrics | `test_agent_task_binding_owner.py`, `test_task_execution_hold_settlement.py`, `test_model_metric_projection_boundary.py` |

The examples are not live provider or production SQL validation. Custom Python
callbacks are not transferable worker state. A configured Metrics recorder
must not alter execution truth, and moving a responsibility does not by itself
change SQL, commits or storage bytes.
