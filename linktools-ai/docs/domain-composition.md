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
| Ordinary Task | `Task`, `TaskGraph`, `TaskNode`, `TaskEngine.open`, `engine.start`, `graph.wait`, `graph.result` | No Agent or Runtime required | TaskEngine borrows Persistence; the outer host closes Persistence |
| Mixed Task graph | `Task.from_agent`, `AgentTaskInput`, `TaskNodeResultRef` | Same Agent handle and same Persistence; explicit result input refs | Task owns scheduling; Agent remains the execution producer; closing Task does not close Agent |
| Evaluation of a Task | `Evaluations.open(engine)`, `publish_dataset`, `start`, `run.wait`, `create_report` | Same ordinary Task producer; no Agent dependency | Evaluations borrows TaskEngine and its Persistence |
| Evaluation of an Agent | `Task.from_agent` plus the same Evaluation entry points | Agent is a target adapter, not a second evaluator | Close evaluation coordination before Task, then Agent |
| Metrics alone | `Metrics.in_memory`, `define`, `record`, `query` | No execution engine or automatic recorder required | Metrics borrows its store and has no producer close; externally owned SQL engines stay host-owned |
| Metrics in Agent/Task | Existing `metrics=provider` argument | Pass the same provider to both domains | Each producer closes only its own recorder, not its peer's recorder or the shared provider |
| One-context composition | `Runtime.open`, `runtime.tasks.bind`, `runtime.evaluations.bind` | One host `async with`; no extra mandatory contexts | Runtime closes composed Evaluation, Task, Agent, then owned Persistence |
| Shared independent domains | `Persistence.open`, `Runtime.attach`, `TaskEngine.open` | Explicit sharing is optional advanced composition | Attached Runtime and TaskEngine borrow the already-open Persistence |

These paths are not claimed to be a completed internal refactor. The example
verifies the call shapes that internal changes must preserve. In particular,
`AgentExecutionPort` is publicly reachable through `Agent.execution_port` and
exported from `runtime.agent_execution`; it is not a disposable private class.
A narrower Task integration should not copy every producer method into a new
forwarding facade just to reduce the number of `service` references.

## Important signatures and semantics

- `Runtime.open(namespace, *, storage, tenant_id="default", context=None,
  features=None, metrics=None, authorization=None, auto_recover=True)` returns
  the context-managed Runtime. Omitting features does not start an Agent
  producer. `Runtime.attach(*, persistence, ...)` borrows an already-open,
  writable Persistence instead. Retain the returned object when the host needs
  to retry cleanup after partial initialization failure.
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
- `Metrics.record(metric, value, *, observation_id=..., ...)` settles an
  explicit source write. Automatic producer recording is a separate recorder
  lifetime. A host can supply a different `MetricRecorderProvider`; a dropped
  metric must not become a different execution outcome.

## Failure, cancellation, and recovery boundaries

A wait timeout or a cancelled observer is not proof that the producer stopped.
Use the graph/execution's explicit `cancel` operation when producer cancellation
is intended; Task's execution handle keeps its Task cancellation settlement.
Preserve the existing graph and execution idempotency keys when resolving an
uncertain outcome instead of blindly starting a new execution.

The existing execution-service protocol exposes
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
recording and dependency-hold acquisition. It receives the original producer
port and the Task services already selected by the engine, rather than a second
set of independent callbacks for the same operations.

The runner captures the compiled declaration and binding once. It reuses the
engine's admission and budget repositories and records the original port for
later reads and recovery. The node executor is resolved lazily because it is
constructed from the bound Task definitions. This does not add another public
adapter, producer, resource owner or user-managed lifecycle.

The existing port's dispatch remains intentional: start/get call the port's
methods, while inspection and retention use its service. These two surfaces
are not required to be the same object. No new identity check or forwarding
API is needed to preserve advanced producer implementations.

## Remaining internal design work

- MCP protocol/client ownership is separate from executable tool Boundary,
  effect facts and recorded response fixtures. Fixture playback creates no live
  MCP client. Stable model naming stays with the execution view, resources
  with Asset/Workspace; do not move fixture authority into the protocol client.
- Task's admission/capture, observation/history, retention and cancellation
  paths still reach multiple parts of AgentExecutionPort. Address their actual
  operation and lifetime contracts together; adding three forwarding methods
  does not complete this boundary.
- Memory and Planning still have domain-specific projection/restore logic in
  Runtime. Preserve Memory's admitted scope and keep model plan tools, execution
  plan mode, and Task scheduling as three different concepts.
- The capability program covers resource capture and admitted Memory reuse;
  crash recovery, custom persistent callbacks and the remaining extension
  paths still require their own evidence. The [complete current Task port
  surface](capability-ownership.md#task-to-agent-complete-current-integration-surface)
  records the contracts that must be preserved before its public design changes.
