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

This association describes JSON values, not model instances. `context.input`
is still an immutable `Mapping[str, JsonValue]`; validate application models
at the callback boundary. `output_type=SomeModel` declares the durable output
schema, and reading that output still returns JSON. Runner-backed Tasks and
references restored from storage retain `JsonValue` unless the application has
the corresponding typed definition. Existing `Task[AppT]` annotations keep that
default. Type parameters add no fields to persisted references or graph identity.

## Conditional batch expansion and collection

The runnable [batch example](../../examples/task_graph.py) uses only public APIs
and an in-memory Runtime, with no model/provider calls:

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

The collector's default `all_succeeded` policy blocks it if a worker fails.
For partial-result reporting, use `all_terminal`, inspect
`context.dependency_states`, and read only successful dependencies. Deliberately
choose `failure_policy="isolate"` on workers if their failure should remain local.
`any_succeeded` also waits until **all** dependencies are terminal; it is not a
race and does not cancel siblings. Scheduling edges alone do not inject data
into model prompts. Agent-backed consumers still need explicit `input_refs`
and their input contract.

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
