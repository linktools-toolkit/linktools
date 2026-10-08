# Shared run budgets

`RunBudget` applies to one root execution and its subagents, or to Runtime-managed
executions in a TaskGraph, including dynamically added nodes. Supply it at admission:

```python
from datetime import datetime, timedelta, timezone
from linktools.ai.core import RunBudget

budget = RunBudget(
    model_requests=20,
    tool_calls=40,
    total_tokens=50_000,
    deadline_at=datetime.now(timezone.utc) + timedelta(minutes=10),
)
execution = await runtime.agents.get("assistant").start("Investigate", budget=budget)
result = (await execution.wait()).result
usage = await execution.budget_usage()

# The same keyword is supported by Session.start/run and TaskEngine.start.
graph_run = await engine.start(graph, idempotency_key="investigation", budget=budget)
graph_usage = await graph_run.budget_usage()
```

Each start creates its own scope. Reusing the same immutable `RunBudget` value in
unrelated starts does not combine their consumption. A graph owns one shared
scope; graph-node executions and their recursive subagents inherit that scope.
The scope identity is persisted, never inferred from correlation fields.
Built-in Agent-backed and callable-task executions inherit automatically.

All dimensions are optional; zero prevents the corresponding new admission.
`deadline_at` must be timezone-aware and is checked against current UTC at each
new effect boundary. Counts are durably admitted before logical SDK model
requests or actual tool handler attempts. Provider transport retries and
continuations within one logical SDK request are not separately counted.
Compaction requests are included. Cached tool effects are not charged again;
a replay-safe tool's new dispatched attempt is counted on its new effect fence.
Capability tools, including subagent delegation, also consume tool calls.

`AgentUsageLimits` still limits an individual Agent run. Both limits apply when
configured. `TaskGraphLimits.max_budget` remains a static sum of node
`budget_cost` weights, independent of these observed-consumption limits.
Custom task functions and expanders receive deadline/token admission checks;
they do not become model or tool calls merely because they are graph nodes.
Direct provider calls or effects outside Runtime-owned boundaries are not
observable or budgeted by this feature.

Custom `Task.from_runner` adapters receive graph deadline/token admission checks
before dispatch, but own the executions and arbitrary effects they create. An
adapter creating a Runtime execution can obtain its admitted scope with
`runtime.tasks.budget_usage(invocation.graph_id, principal=invocation.principal)`
and pass the returned `scope_id` as `budget_scope_id` to the existing
`ExecutionService.start` or `start_task` operation. The service authorizes the
owner and validates the durable scope. Supplying both a fresh `budget` and an
inherited scope is invalid. Without that explicit propagation, extra executions
are independent; the graph counters do not claim to cover them. Direct provider
calls in ordinary callable bodies have the same instrumentation limitation.

## Tokens and unknown usage

Token limits are **soft observed thresholds**, preserving graph concurrency.
New model/tool effects are refused once settled usage reaches the threshold.
Calls already admitted may finish, so concurrent provider consumption can
exceed the threshold without a fixed numeric overshoot bound. No token
estimation, numeric in-flight token reservation, monetary limit, or external
billing guarantee is implied.

`BudgetUsage.total_tokens` is the known settled subtotal of SDK-reported input
plus output tokens for individual requests. Cache counters are already part of
input usage and are not added again. It is not aggregate Agent `RunUsage` and
never includes a parent's descendant aggregate a second time. The SDK may
represent omitted provider usage as zero in a successful response; this
feature cannot distinguish that from a reported zero without provider usage
provenance.

`in_flight_model_requests` and `unknown_model_requests` distinguish active
reservations from terminal calls whose usage was not available. A failure or
cancellation after admission retains the count and records unknown tokens;
cancellation is not a refund. A known subtotal of zero with unknown calls does
not mean those calls consumed zero tokens.

In-flight requests in the same Runtime may run concurrently. Terminal unknown
usage, or an unresolved reservation from a previous Runtime/process, blocks new
effects in token-limited scopes. The refusal uses
`EXECUTION_USAGE_LIMIT_EXCEEDED` with `safe_details.reason` of `unknown_usage` or
`unresolved_in_flight`, distinct from `limit_exceeded` and `deadline_exceeded`.
There is no automatic timeout release or guessed settlement. Count-only scopes
can continue within their remaining admitted-call limits. Duplicate dispatch
identities are refused; repeated identical settlement is idempotent.

## Continuation, restart, and retention

- Deferred continuation and recovery keep the same execution and scope
- Retry and default fork inherit the original scope and consumed allowance
- An explicit `execution.fork(..., budget=RunBudget(...))` creates an independent
  scope. This is an intentional new allowance, not reconciliation or refund of
  unknown consumption in the old scope
- Starting a new Session turn is a separate root execution and accepts its own
  budget; budgets are not silently shared across the entire Session
- Reusing a start idempotency key with changed limits conflicts
- Expiration/exhaustion stops new admissions and retains completed results. It
  does not cancel in-flight calls or revoke an already-started external effect

Memory, filesystem, and SQL-backed Runtime state use the existing transaction
and optimistic-concurrency owner for scope and reservation updates. Durable
backends retain admitted counts, settled tokens, and unresolved reservations
across restart. A missing scope is an integrity failure, not a new empty budget.
Cross-process token admission is conservative: a process cannot assume another
process's outstanding request has known usage, so it refuses that scope rather
than pretending to provide a distributed in-flight usage oracle. Atomic count
admission does not establish exactly-once execution at external providers.

Scope summaries and receipts are retained independently of individual execution
cleanup because graphs, retries, and forks can continue to reference them.
This first implementation does not automatically garbage-collect shared budget
scopes or offer a public provider-billing reconciliation API. Plan durable
storage capacity accordingly. Runtime snapshots include and validate their
reservation-derived projections; restoring an unresolved reservation keeps it
unresolved. Existing no-budget calls do not create budget records or acquire
budget admission permits.
