# Runtime entry points and bound handles

The [complete offline program](../examples/runtime_entrypoints.py) includes
imports, model configuration, Agent and Session starts, result reads, Task-only
execution, cancellation, durable handoff/recovery and resource closing. It calls
the real public APIs without a model provider account. Run it after the normal
editable installation:

```bash
python linktools-ai/examples/runtime_entrypoints.py
```

## One handle for each entity

Runtime composes domains. An `Agent` describes how to start work, a `Session`
binds an Agent to a conversation ID, and an `Execution` identifies admitted work.
No caller needs to construct a Conversation, a Store, an adapter or a second
manager for ordinary Agent calls.

| Entry | Previous return | Current return and responsibility |
| --- | --- | --- |
| `Runtime.open(...)`, `Runtime.attach(...)` | `Runtime` | Unchanged: owned or borrowed composition and lifecycle |
| `runtime.agents.get(agent_id="default")`, `.restore(binding)` | `Agent` | Unchanged: live or captured definition handle |
| `agent.start(prompt, ...)`, `session.start(prompt, ...)` | `Execution` | Unchanged: accepted execution handle |
| `agent.run(...)`, `session.run(...)` | `WaitResult[ExecutionResult]` | Unchanged: start and wait, with optional observation |
| `agent.session(id, ...)` | `Session` | Unchanged: lazy definition-bound handle, no creation/read |
| `agent.create_session(id, ...)` | `Session` | Unchanged: create then bind, retaining Agent derivations |
| `runtime.sessions.get(id, principal=...)` | `Session` | Unchanged: authorized lookup and handle |
| `runtime.sessions.create(agent_id, request)` | `SessionView` | `Session`, bound to `request.principal` |
| `runtime.sessions.fork(agent_id, id, request)` | `SessionView` | `Session`, bound to `request.principal` |
| `runtime.sessions.inspect(id, principal=...)` | No facade operation | `SessionView`, the explicit state-read operation |
| `runtime.executions.get(id, principal=...)` | `Execution` | Unchanged: authorized lookup and handle |
| `runtime.executions.retry(id, request)`, `.fork(id, request)` | `ExecutionHandle` receipt | `Execution`, bound to `request.principal` |
| `runtime.executions.recover(id, principal=..., idempotency_key=None)` | `ExecutionHandle` receipt | `Execution` for the original ID, bound to the recovery actor |
| `execution.inspect()`, `.result()` | Only ID-based reads | Current `ExecutionView` or `ExecutionResult`, using the bound actor |
| `execution.principal`, `session.principal` | Private bound actor | Read-only actor metadata |

ID-based APIs retain explicit request DTOs and principals. Convenience handles
supply their bound principal. `Execution.result()` reads the current result;
`Execution.wait()` waits and returns `WaitResult[ExecutionResult]`. The former is
not a synonym for waiting. `inspect` and `list` return views rather than handles.
Session `update`, `close`, `reconcile` and `wait_inactive` still return their
operation's `SessionView`; `refresh` returns `SessionDetail`.

Creation, fork, retry and recovery bind their returned handle directly from the
successful operation's receipt/view. They do not perform a second storage read
or authorization call after admission. A later `get` or `inspect` is an explicit
fresh observation. Lower-level producer services still use `ExecutionHandle` as
an admission receipt; the Runtime facade presents the richer entity handle.
There is no compatibility alias or data migration for the superseded facade
returns.

Hosts directly assembling the exported facades supply the new keyword-only
`bind_execution: Callable[[str, Principal], Execution]` or
`bind_session: Callable[[SessionView, Principal], Session]` alongside the existing
async lookup callback. These synchronous constructors consume the already
accepted outcome and must not start another operation. Runtime supplies them;
ordinary callers still obtain these facades from Runtime.

## Agent and Session convenience paths

The ordinary chain stays:

```python
agent = runtime.agents.get()
execution = await agent.start("Hello")
outcome = await execution.wait()
result = await execution.result()
```

`Agent.start` defaults to run mode, an automatically generated idempotency key,
the Runtime principal, no files, no session and the current Agent's planning and
thinking defaults. `session.start` binds `session_id`; it still delegates to the
same Agent producer. `Session.principal` resolves to the Runtime principal when
its lazy handle was created without an explicit principal. Neither convenience
path requires an explicitly created Session.

For a stored Session use `sessions.create/get`; for a definition-bound lazy
handle use `agent.session`. The latter keeps derived or restored Agent semantics
and deliberately does not check that a Session exists. `Session.fork` and
`Agent.create_session` preserve those definition bindings rather than replacing
them with a fresh default Agent.

## Task and graph semantics stay with Task

`Runtime.tasks.bind(*definitions) -> TaskEngine` composes an owned Task producer.
`TaskEngine.open(*definitions, persistence=...)` instead creates an independently
owned producer borrowing an already open Persistence. Both paths use:

```python
run = await engine.start(graph, principal=principal, idempotency_key="start")
run = await engine.get(run.graph_id, principal=principal)
node_execution = await run.execution("node")
result = await node_execution.result()
await node_execution.cancel(idempotency_key="cancel-node")
```

The node handle is the same `Execution` type for plain Python and Agent nodes.
Its cancellation operation also settles the node/graph cancellation obligation.
A direct Agent `runtime.executions.cancel(id, request)` does not substitute for
that graph operation. Graph recovery remains `run.recover(...)`, returning the
`TaskGraphResult` of the graph-control operation, like graph cancellation and
input resumption. An individual execution recovery cannot replace dependency,
node-admission and graph-recovery validation.

`Runtime.executions` is the configured Agent producer's facade. A Task-only
Runtime intentionally does not initialize that producer just to offer the same
property. Task lookup stays `engine.get(graph_id).execution(node_id)`. An execution
ID does not identify which of several TaskEngine instances owns graph control;
scanning engines or picking an arbitrary same-definition producer would silently
change cancellation and recovery semantics. The distinct graph path is a real
ownership boundary, not a second execution implementation.

## Task-to-Agent operations

`AgentExecutionPort` is the existing producer boundary, not a new adapter.
Task consumers call only operations they use:

- Definition capture/restoration, same-Persistence/open-producer validation,
  file canonicalization and input materialization.
- `start` and `get`, including a graph-aware cancel callback when constructing a
  Task-owned node handle.
- Authorized inspect/result/wait, result size, children and budget usage.
- Cancellation, dependency-hold acquisition/release and terminal handoff.
- Model-observation cutoff, update, metadata and subscription operations.
- Invocation capture and graph-authorized recovery of a bound Agent execution.

The compiler, producer service, handle-operation binding, sessions and input
capture implementation are private dependencies of that port. Task no longer
reaches through `port.service` or `port.operations`. Reads and retention still
use the actual producer service, which need not be the same object as the handle
binding's service. Model observation keeps its history owner. This does not
redirect retention operations to a different override or drop `terminal_commit`.

## Errors, identity and lifecycle

- Existing errors and unknown outcomes propagate unchanged. An uncertain start
  must be resolved with the original idempotency key; constructing a handle does
  not convert it into success or start new work.
- The handle principal authorizes its operations. A recovery actor never replaces
  the principal recorded when the execution or TaskGraph was admitted.
- Normal Runtime close drains its owned producers and cancels unfinished work.
  `handoff()` preserves eligible checkpointed Agent executions for explicit
  recovery in a fresh Runtime. Repeated close and failed-close retry semantics
  are unchanged.
- `Runtime.open` owns Persistence. `Runtime.attach` borrows it. Independently
  opened TaskEngine/Evaluations instances retain their own close responsibility;
  Runtime-owned `tasks.bind`/`evaluations.bind` instances close with Runtime.
- `History.open` remains independent. An explicitly supplied native StepStore
  remains caller-owned; this API change neither replaces it nor adds another
  required context manager.

## Complete composition and coverage map

The small program above covers the common handle paths. The existing complete
[domain composition program](../examples/domain_composition.py) additionally
shows Task-only evaluation, Agent/mixed DAGs sharing Persistence, Runtime-owned
Agent evaluation, independent/composed Metrics, Checkpoint/Trace and closing.
[Capability ownership](capability-ownership.md) covers declarations, live/native
capabilities, binding and restore. [MCP usage](mcp-protocol-owner.md) covers the
protocol/client lifetime; these owners are not reimplemented by this interface
change.

| Contract | Coverage |
| --- | --- |
| Agent/Session create/get/fork, execution retry/fork, principal and state/result reads | `test_runtime_entity_handles.py` |
| Successful admission binds without a follow-up read; unknown/error remains unchanged | `test_runtime_entity_handles.py` |
| Handoff/recover returns Execution and retains admitted principal | `test_runtime_entity_handles.py`, `test_runtime_shutdown_handoff.py`, `test_graph_bound_recovery.py` |
| Plain Task and Agent-node graph-aware cancellation; Task-only needs no Agent | `test_runtime_entity_handles.py`, `test_task_mixed_node_reliability.py` |
| Producer dispatch distinct from handle service, snapshot capture, input capture failure/cancellation, holds and history | `test_agent_task_binding_owner.py`, `test_task_execution_read_routing.py`, `test_task_input_projection_recovery.py` |
| Observation lifecycle and shared waits | `test_unified_wait_contract.py`, `test_runtime_watch_api.py` |
| Owned/borrowed resources, partial initialization and custom StepStore | `test_runtime_resource_ownership.py`, `test_persistence_step_composition.py`, `test_runtime_composition_invariants.py` |

These are coverage locations, not a claim that checks have run for a particular
commit.
