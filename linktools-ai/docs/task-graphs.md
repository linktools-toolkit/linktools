# Authoring Task graphs

Keep named `Task` definitions, frozen node inputs, and scheduling dependencies
separate. Use ordinary Python to choose nodes and the existing `TaskExpander`
for work discovered during execution. The durable scheduler owns execution,
recovery, cancellation, and observation.

## Typed JSON results

A function Task can retain its JSON result type through an authoring reference:

```python
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext, TaskNodeResultRef


async def prepare(context: TaskNodeContext[None]) -> str:
    question = context.input["question"]
    if not isinstance(question, str):
        raise ValueError("question must be text")
    return question.strip()


prepare_task = Task("example.prepare", prepare, effect_policy="none")
prepare_node = TaskNode("prepare", task=prepare_task, input={"question": " hello "})
prepared = TaskNodeResultRef(prepare_node)


async def finish(context: TaskNodeContext[None]) -> str:
    text = await context.read_dependency(prepared)  # inferred as str
    return text.upper()


finish_task = Task("example.finish", finish, effect_policy="none")
graph = TaskGraph("example", (
    prepare_node,
    TaskNode("finish", (prepare_node.node_id,), task=finish_task),
))
```

After binding both Tasks and running the graph, `await run.result(prepared)`
is also inferred as `str`; `await run.result_ref(prepared)` is
`TaskResultRef[str]`. A reference selects an already declared dependency. It
does not create an edge or bypass result authorization, terminal-status checks,
or digest verification.

The old string forms remain useful for dynamic graphs and input aliases:
`TaskNodeResultRef("prepare")`, `context.read_dependency("prepare")`, and
`run.result("prepare")` return the general `JsonValue` type. For example, a
consumer that previously needed `isinstance(value, str)` after reading a known
producer can instead retain `TaskNodeResultRef(prepare_node)` and let the type
checker validate operations on the result. There is no extra decode method.

Container results such as `list[str]`, `dict[str, int]`, and nested JSON
collections retain their types too. The result type parameter is deliberately
unbounded: a recursive mutable `JsonValue` bound would reject these valid
containers through invariance. Runtime JSON validation still rejects unsupported
Python objects; a type annotation does not change serialization. Heterogeneous
graphs and definition registries erase result types while each retained typed
reference preserves its producer association.

This association describes JSON values, not model instances. `context.input`
is still an immutable `Mapping[str, JsonValue]`; validate application models
at the callback boundary. `output_type=SomeModel` declares the durable output
schema, and reading that output still returns JSON. Runner-backed Tasks and
references restored from storage retain `JsonValue` unless the application has
the corresponding typed definition. Existing `Task[AppT]` annotations keep that
default. Type parameters add no fields to persisted references or graph identity.

## Agent input projection recovery

`Task.from_agent(..., build_input=...)` projects declared, durable predecessor
results before preparing the consumer prompt. A temporary callback failure stops
the original graph in `RECOVERY_REQUIRED` with error code
`TASK_INPUT_PROJECTION_FAILED`, phase `task_input_projection`, and a safe
`cause_code`. After the dependency becomes available, reopen the same persistence,
bind the same Task definitions, obtain the original run, and call
`await run.recover(idempotency_key=...)`. Successful predecessors keep their
execution IDs and results; recovery does not call their models again. A sustained
outage can be recovered explicitly again without resubmitting the graph.

Only retryable availability errors (`STORAGE_UNAVAILABLE`,
`STORAGE_DEPENDENCY_NOT_READY`, `RUNTIME_DEPENDENCY_NOT_READY`,
`SERVICE_NOT_READY`, `MODEL_UNAVAILABLE`), `MODEL_RATE_LIMITED`, `MODEL_TIMEOUT`,
and `WAIT_TIMEOUT` qualify at this callback boundary. A retryable flag alone
does not make schema, authorization, integrity, owner or fence errors temporary.
Other projection errors fail normally. Existing terminal `FAILED` graphs are
not reopened by this behavior.

Projection callbacks must be read-only or idempotent under a fixed application
identity: a crash after the callback returns but before its prepared input is
published can replay the callback. Once prepared input is durably published,
recovery restores that prompt and its source references instead of projecting
again. A bound or uncertain consumer execution follows the existing execution
recovery rules; it does not authorize another projection. Cancellation and lease
fences continue to govern recovery and publication.

## Custom Task persistence and request identity

Custom `TaskPersistence.admissions` implementations must provide
`get_submission_ref(graph_id, *, tenant_id) -> TaskSubmissionRef | None`.
The lookup returns the retained submission identity for prepared, admitted,
and cancelled graphs. It is required, not an optional capability. Reading only
committed admissions is insufficient: cancellation before admission removes
the prepared graph payload but retains the submission reference.

`TaskGraphAdmission.request_identity_version` owns the comparison rule;
`TaskSubmissionRef` carries a verified projection of that field with the graph,
principal, operation ID, and request digest. Retain the same version through
preparation, admission, cancellation, recovery, and commit readback. Reject
inconsistent admission/reference pairs. Older durable records that omit the
field mean version 1; new submissions use version 2, including direct repository
prepare/admit calls. An existing owner's version is selected before hashing.
A concurrent winner must be read back and compared using that one version.
Cancellation can establish a new fence only from a version-2 reference. An old
version-1 reference that was never durably prepared cannot be upgraded from its
digest alone and is rejected; cancellation of retained version-1 submissions
continues to use their original identity.

Version 2 preserves StructuredContent's value, rendered content, tag,
description, and metadata. Existing version-1 Agent submissions keep their
previous text-content comparison, without reconstructing discarded structured
values or rewriting old records. Only Agent-owned prompt contracts receive
this projection; ordinary function and runner JSON remains unchanged. Keep the
original graph until preparation finishes so a competing durable owner can be
compared without recapturing workspace files or hashing a materialized prompt.

StructuredContent values in Agent and Task requests, input captures, and external
results follow Pydantic's native JSON serialization, including custom serializers,
aliases, and model configuration. JSON object key order is immaterial; JSON array
order is part of identity. Duplicate serialized object keys are rejected rather
than silently dropping a value. Python sets and frozensets can produce different
array orders across processes, so the same logical set can conflict when retrying
a version-2 request or result after restart. Use an explicitly ordered JSON value
or a deterministic Pydantic serializer, such as one returning a sorted list, when
stable retries are required. Custom serializers must also return stable values
for those retries.

## Conditional batch expansion and collection

The runnable [batch example](../../examples/task_graph.py) uses only public APIs
and a Runtime-bound TaskEngine with in-memory storage, with no model/provider calls:

```bash
python examples/task_graph.py
```

Its plan Task produces frozen JSON describing the items and chosen operation.
The expander selects one existing Task and declares one node per item, followed
by a collector depending on those nodes:

```python
def expand(context: TaskExpansionContext) -> tuple[TaskNode, ...]:
    value = context.output
    assert isinstance(value, dict)
    texts = value["texts"]
    assert isinstance(texts, list)
    worker = uppercase_task if value["uppercase"] else lowercase_task
    nodes = tuple(
        TaskNode(f"item-{index:04d}", task=worker, input={"text": text})
        for index, text in enumerate(texts)
    )
    return (*nodes, TaskNode("collect", tuple(node.node_id for node in nodes), task=collect_task))
```

Bind both possible Tasks and the expander before submission. Node IDs must be
unique across the whole graph; when expanding multiple batches, include a
stable source/batch identity in their IDs. Keep the expander deterministic for
the same source input/output, within the graph's node/depth/budget limits.
An empty batch still emits the collector, which returns an empty list. The
example sorts zero-padded item IDs to preserve input order, independent of
completion order.

This is the same composition shape as planning audit workers and then publishing
their combined result, or choosing an evaluation path and collecting its outputs.
Domain-specific routing and result models belong to the caller. A shared helper
is useful only when independent real callers repeat the same operation.

## Join membership and terminal policies

A join is an ordinary TaskNode whose declared dependencies identify its members.
For dynamic work, the expander can return workers and their collector together,
with the collector depending on exactly those selected worker IDs. Unselected
branches are absent; they are not persisted as a separate `SKIPPED` status.

| Dependency policy | When the node can run | No dependencies |
| --- | --- | --- |
| `all_succeeded` | Every dependency succeeded; a failed, blocked, or cancelled dependency blocks the node | Ready |
| `all_terminal` | Every dependency is terminal, regardless of success | Ready |
| `any_succeeded` | Every dependency is terminal and at least one succeeded | Blocked |

`SUCCEEDED`, `FAILED`, `BLOCKED`, and `CANCELLED` are terminal. `WAITING` and
`RECOVERY_REQUIRED` are not. `any_succeeded` is an all-terminal barrier, not an
early winner: it never cancels siblings. An empty worker batch can still emit
an empty collector, but a source that fails before expansion is not a successful
empty batch.

For partial-result reporting, use `all_terminal`, inspect
`context.dependency_states`, and read only successful dependencies. A successful
JSON `null` remains a result; a failed or cancelled dependency is not converted
to `null`. Choose `failure_policy="isolate"` on workers whose `FAILED` or
`BLOCKED` status should not propagate to the final graph status. This does not
change those nodes' statuses or the collector's dependency policy. `CANCELLED`
nodes can still make the graph `CANCELLED`; cancellation is not isolated.

The launcher prepares dependency metadata through
`TaskRepository.get_dependency_records(graph_id, node_ids, tenant_id=...)`.
The repository reads only the requested node states/definitions and their
successful result records, in one read transaction with bounded key batches.
It returns states in request order and result metadata keyed by node ID; result
bodies remain lazy. A mismatched tenant returns empty collections. Missing
state/definition records or invalid stored data remain integrity errors; the
launcher also rejects missing or inconsistent successful result metadata and
rechecks the node's terminal dependency policy. Backend read isolation is
unchanged; no graph-wide cache or additional durable state is introduced.

### Nested expansion is not an implicit subtree barrier

If `branch` expands a `grandchild`, a collector depending only on `branch` can
run once `branch` succeeds, even while `grandchild` is still running. The whole
graph remains unsettled until its nodes settle. Declare the actual nodes that a
collector must wait for when constructing its expansion. Observe the whole graph
with the existing run `wait()` operation. Inspect `outcome.result.wait_status`:
`WAITING` and `RECOVERY_REQUIRED` can stop that wait without terminal completion.

A scoped descendant join would require an explicit membership and closure
contract that survives cancellation and recovery. It is not part of the current
API. Do not approximate it with all graph nodes or an Execution parent/subagent
tree: those are different relationships. Existing direct barriers keep their
meaning after nested expansion and restart.

Scheduling edges alone do not inject data into model prompts. Agent-backed
consumers still need explicit `input_refs` and their input contract. Collect
results through the existing authorized readers; keep application-specific
failure conversion and reduction order in the consumer.

## Comparison choices

The useful ideas from pydantic-graph's
[typed steps](https://github.com/pydantic/pydantic-ai/blob/v2.55.0/pydantic_graph/pydantic_graph/step.py)
and [fork-scoped joins](https://github.com/pydantic/pydantic-ai/blob/v2.54.0/pydantic_graph/pydantic_graph/graph_builder.py)
are evaluated against this durable execution model:

| Capability | LinkTools choice |
| --- | --- |
| Typed step results | Preserve JSON result types on existing definitions and references, including collections |
| Decisions and mapped work | Select nodes in ordinary Python; admit dynamic work with TaskExpander |
| Join and collection | Keep explicit dependency membership, the three terminal policies, and authorized result reads |
| Nested fork scope / early winner | Not added to direct dependencies; neither recursive waiting nor sibling cancellation is implicit |
| Definition diagnostics | Reuse TaskGraph validation and describe_submission |
| Graph display and execution state | Reuse existing read-only projections and the durable scheduler; no parallel graph state or shared mutable State framework |

A new scoped join remains a possible future capability when a concrete consumer
needs it. It is not required to express the existing flat expansion and
collection pattern, and must not silently change existing graph behavior.

## Diagnose before submission

Use the same bound engine for inspection and execution:

```python
description = await engine.describe_submission(graph, idempotency_key="request-1")
run = await engine.start(graph, idempotency_key="request-1")
```

`describe_submission()` resolves the currently declared nodes without submitting
the graph or executing Tasks. It uses the normal admission checks, so invalid
normalized input and incompatible output contracts report their `graph_id` and
`node_id` in `AIError.safe_details`. These diagnostics do not include input
bodies. Dynamically expanded nodes are checked when expansion admits them;
inspection cannot predict nodes produced by future execution.

Use the existing [watch/wait and graph projection APIs](task-observation.md) to
observe a run, including dynamically admitted nodes. Use
[runtime history](runtime-history.md) for paginated retained records and the
[evaluation guide](evaluation.md#native-graph-candidates) for graph candidates.

## Observation and persistence IO

`TaskEngine(..., model_refresh_interval_seconds=5.0)` and `TaskEngine.open`
accept a finite positive fallback period for model metadata in graph `watch`
and observed `wait`. The default remains 1 second. Definition views and
handles obtained through `get` keep the engine's period. Increasing it reduces
idle metadata reads when many executions are observed; remote metadata without
local notifications can take up to that period to appear. Local notifications
still wake immediately, and initial replay and the final authoritative
projection do not wait for the timer. This is an observation preference, not
part of the admitted graph's durable identity.

The setting does not change scheduler cadence, lease renewal, execution-tree
discovery or durable event polling. Task and Execution event streams already
use local notifications and a 1-to-30-second idle fallback backoff. Never use
an observation interval to delay claims, execution bindings, handoffs,
cancellation or terminal commits.

Reuse a long-lived `Persistence` and its storage routes across producers.
SQL routes sharing an `AsyncEngine` share its connection pool; a session or
transaction scope is not a new physical connection. Configure server pool
size, overflow and recycling on the caller-owned SQLAlchemy engine and dispose
it only after borrowed producers and Persistence have closed. SQLite storage
also retains its engine. A close-free `SQLiteMetricStore`, by contrast, uses
operation-scoped engines; use a caller-owned `SqlMetricStore` engine when
Metrics needs explicit pooling and lifetime control.

For optional Agent trace/capability writes, use the existing
`CommitPolicy.batch(batch_size=..., flush_interval=..., buffer=...)` on the
owning capability. A batch policy without a buffer publishes synchronously;
there is no hidden queue. Trace defaults are 128 entries and 3 seconds when a
buffer is supplied. Required flushes, terminal publication, cancellation
settlement and unknown-commit readback retain their owning safety boundaries.
See [capability persistence](capability-persistence.md) for acceptance versus
durability and failure semantics. These settings do not batch arbitrary Task
state transitions across recovery boundaries.

Measure SQL statements, transactions, physical connects and pooled checkouts
separately. None is a disk-fsync or wire-byte measurement. The matched idle
observer test `test_task_observation_interval.py` compares the default period
with 10 seconds over the same blocked execution and verifies both final
projections. The connection/batch probes in `test_task_io_controls.py` isolate
existing control behavior rather than claiming those mechanisms are new.

To reproduce the focused checks from an installed development environment:

```sh
PYTHONPATH="$PWD:$PWD/linktools/src:$PWD/linktools-ai/src" \
PYTEST_ADDOPTS='-k "task_observation_interval or task_io_controls or graph_metadata_projection" -s' \
python manage.py check linktools-ai --skip-compatibility
```

A matched SQLite run with SQLAlchemy 2.1.8 measured 10 independent read
transactions: `NullPool` created 10 physical connections while the default
pool created 1; both performed 10 checkouts and 10 transactions. For eight
Trace observations plus an explicit final flush, source commits were 26
without a buffer and 5 with a buffer; both retained all eight observations.
These are configuration comparisons of existing mechanisms, not reductions
introduced by the observation-interval parameter, and are not server network
or physical disk measurements. Exact totals are diagnostic evidence rather
than frozen backend contracts.

In the same probe environment, two independent observer engines watched one
blocked Agent execution after initial replay. Over a 3.2-second idle window,
the default 1-second refresh performed 29 SELECT statements and 18 logical
transactions; a 10-second refresh performed 13 SELECTs and 11 transactions.
Both observers completed their final projection within the test's 5-second
completion bound, without waiting for the 10-second timer. These counts include
other observer polling, so the longer interval does not eliminate all reads.
Real event-loop scheduling and shared-pool contention can vary exact totals.
