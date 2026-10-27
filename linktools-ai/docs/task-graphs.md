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
