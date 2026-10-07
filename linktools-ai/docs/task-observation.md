# Watching and waiting for Runtime operations

`Execution`, `TaskGraphRun`, and `EvaluationRun` expose `watch()` and `wait()`.
Watch is an async iterator. Wait optionally delivers events to `on_event` while
waiting for authoritative state, and always returns `WaitResult`:

```python
from linktools.ai.runtime import TaskGraphRunEvent

async def publish(event: TaskGraphRunEvent) -> None:
    await projection.apply_idempotently(event)

outcome = await run.wait(
    on_event=publish,
    cursor=saved_cursor,
    include_event_content=False,
    timeout_seconds=120.0,
    close_timeout_seconds=5.0,
)
await projection.refresh_graph(outcome.result, status=outcome.result.wait_status)
await projection.save_cursor(outcome.cursor)
if outcome.observation_error is not None:
    await projection.show_observation_gap()

# No callback: the result container is unchanged, and no stream is started.
outcome = await execution.wait()
output = outcome.result.output

# Observation only: save each cursor after successful consumption.
async for item in run.watch(cursor=saved_cursor, include_content=False):
    await publish(item)
    saved_cursor = item.cursor
```

The projection methods above belong to the application. RuntimeTaskGraphExecutor
and its UI/storage implementation are outside this repository.

## Results and stopping states

`WaitResult[T]` contains only `result`, `cursor`, and `observation_error`.
The result never changes shape based on whether `on_event` is supplied:

- Execution and RuntimeExecutions.wait return `WaitResult[ExecutionResult]`
- Agent.run/plan and Session.run/plan forward the same observation options and
  return `WaitResult[ExecutionResult]`, preserving diagnostics and cursors
- TaskGraphRun.wait returns `WaitResult[TaskGraphInfo]` by default, or
  `WaitResult[TaskGraphState]` with `include_content=True`. Both come from the
  same stopping service read. No second state read reconstructs the result
- EvaluationRun.wait returns `WaitResult[EvaluationView]`

Execution stops at SUCCEEDED, FAILED, or CANCELLED, and preserves local terminal
worker cleanup/failure checking. WAITING_DEFERRED continues waiting;
RECOVERY_REQUIRED raises its recovery error. Failed and cancelled results may
return normally without output.

Graph terminal states, RECOVERY_REQUIRED, and stable WAITING stop the wait.
`result.status` is the durable raw status. `result.wait_status` also projects raw
RUNNING to WAITING when unfinished nodes cannot run and at least one is waiting.
This is a computed property, not persisted state. `result.node_states` replaces
old wait payloads' `node_results`; node output content remains in
`results()`, `result()`, and `result_ref()`.

The returned graph includes all current nodes, including dynamically expanded
ones. `await run.state()` reads safe current metadata; use
`await run.state(include_content=True)` for raw node definitions and invocation
inputs together with their current state. This replaces the redundant SDK
`inspect()` read. It is a fresh snapshot, not the original submitted definition
or necessarily the same stopping snapshot returned by wait.

Evaluation stops at complete, cancelled, or needs_attention. A return is not
necessarily business success. Ordinary human scoring may remain running while a
graph is WAITING, until a score is supplied or the wait times out.

Without a callback, cursor must be None. With a callback, cursor and
include_event_content identify the same observation scope as watch's
include_content. Graph wait's separate include_content controls only the result;
it does not change cursor compatibility or callback payloads. New Agent/Session
run/plan calls do not accept a cursor: resume observation on the existing handle.

## Delivery, preparation, and cleanup

Callbacks run serially. A successful callback return acknowledges its item
cursor. Failure, cancellation, or unfinished delivery keeps the previous ACK;
if no item succeeds, the input cursor is preserved. Durable sequence watermarks
advance only for durable events. `ExecutionEvent.event_seq` and
`TaskEvent.event_seq` are one-based within their execution or graph. Execution
stream events expose this coordinate as `durable_seq`; live deltas have
`durable_seq=None` and do not advance durable watermarks. Low-level event
readers resume with `after_event_seq`; execution-tree readers use the
per-execution `after_event_seqs` mapping.
External side effects require an idempotent consumer; delivery is at least once.

Cursors are opaque and bind namespace, tenant, operation identity, observation
kind, and content mode. Graph, execution, and evaluation cursors are not
interchangeable. Owner and known resumed-member validation completes before an
already-finished authoritative waiter may short-circuit observation. Dynamic
members are validated when discovered.

Authoritative completion may close the observation before all final events, or
any events, have been delivered. Wait does not promise final drain. Refresh the
UI from the authoritative result and resume watch from the returned cursor to
read remaining durable events. Stream EOF alone does not finish a wait.

Timeout must be a finite nonnegative number or None; close_timeout_seconds must
be finite and positive. Booleans, NaN, infinity, non-callable on_event, invalid
content mode, and cursor without a callback are rejected before tasks start.
Agent/Session convenience methods validate these options before admission.

SDK deadlines include observation preparation and raise `AIError(WAIT_TIMEOUT)`
with scope, resource_id, and last ACK cursor. The deadline begins at the bound
handle wait; Agent/Session start and facade get are outside it. Raw services keep
their own timeout contracts and do not receive user callbacks.

Only explicitly identified optional presentation transport failures degrade to
`observation_error`. Authorization, cursor, lineage, durable read, codec and
integrity failures propagate. A callback's ordinary exception, including AIError
or TimeoutError, becomes `ObservationError(origin="callback")` with its cause
and previous ACK. Cancellation is not translated.

Simultaneous failure priority is cancellation, authoritative waiter failure,
watch contract failure, callback failure, SDK deadline, then cleanup failure.
Local cleanup has its own close budget. Runtime retains unfinished tasks and
rejects new waits while closing; Runtime.close cancels all sessions before
waiting for any one to drain, and can be retried if noncooperative tasks remain.
A cleanup timeout is not successful completion. It reports phase="cleanup" and
cleanup_pending=True unless a higher-priority error already wins.

Timeout plus cleanup may take both budgets. Blocking synchronous code and
cancellation-resistant coroutines cannot be forcibly stopped by Python; the
bounds require a schedulable event loop. No automatic thread/process is created.
Waiting, watching, timeout and caller cancellation do not cancel durable work.

## Recursive execution trees and evaluation graphs

Execution.watch observes the selected execution and arbitrarily deep descendants.
A SUBAGENT may itself be selected as the observation root. Depth is relative to
that selected root; real parent/root/lineage metadata is preserved. Graph.watch
uses the same tree observation for its agent node bindings, including dynamic
and recovered nodes. Order is guaranteed per execution, not globally across
independent streams. N discovered executions require O(N) streams and cursor
entries; cross-process discovery polls durable child records.

After all current streams reach EOF, one final discovery pass drains members
found then. Membership established after that pass belongs to a later watch.
EOF is an observation-phase boundary, not proof that admission permanently
closed the execution subtree. Live deltas are not persisted and cannot be
recovered across a disconnect; durable facts and existing content APIs remain
available.

Evaluation.watch yields existing TaskGraphRunEvent objects from confirmed
intents, including released graphs for history. Unconfirmed intents remain
invisible until confirmation; watching never admits, ticks, or reconciles them.
Target-to-scorer gaps do not end evaluation observation. A known graph that
resumes after a WAITING EOF is reopened from its delivered cursor when its
durable graph watermark advances.

An evaluation cursor wraps a graph-id-to-graph-cursor map. Even child cleanup
errors report the evaluation cursor. Complete/cancelled/needs_attention capture
final confirmed members and finite per-stream durable cutoffs, then drain from
the delivered watermarks without waiting for RUNNING/WAITING child streams to
end. These vector cutoffs are not a cross-stream atomic snapshot. A later
reconcile can be observed with another watch; rescore has a separate identity.
Evaluation wait may stop before this standalone-watch drain finishes.

TaskGraphRun.replay remains a finite historical replay plus result read, not an
alias for watch. Content/history pagination remains a separate capability:
ordinary history/trace/transcript preserve their existing scope; recursive tree
watch does not silently change their ordering or paging contracts.
For immediate same-Runtime content readback and event locators, see
[Runtime history](runtime-history.md).

## Cancellation and results paging

Use a stable idempotency key for one explicit cancellation intent:

```python
await run.cancel(idempotency_key=saved_cancel_key, force=False)
```

Control may commit after its caller times out or is cancelled. Read state back:
CANCELLED confirms cancellation; another terminal state confirms that outcome;
nonterminal state requires reconciliation with the same key; a failed readback
means outcome unknown. Never infer non-commit or silently enable force.

Graph results pages bind a graph sequence. A within-page change raises
STORAGE_CONFLICT; a between-page sequence change raises CURSOR_INVALID. Discard
unpublished partial pages and restart, with a bounded application retry budget.
Publish only after next_cursor is None. Respect content_included and byte limits;
a result page and the earlier wait result are not a cross-store transaction.

For example, collect a complete page set before publishing any of it:

```python
from linktools.ai.errors import AIError, ErrorCode

for attempt in range(3):
    items = []
    cursor = None
    try:
        while True:
            page = await run.results(cursor=cursor, include_content=True)
            items.extend(page.items)
            cursor = page.next_cursor
            if cursor is None:
                break
    except AIError as error:
        if error.code not in {ErrorCode.CURSOR_INVALID, ErrorCode.STORAGE_CONFLICT}:
            raise
        if attempt == 2:
            raise  # Leave the previous published view intact.
    else:
        await publish_results(items)  # Application-owned publication.
        break
```

This retry applies to graph results, not every `Page`: history cursors continue
a fixed high-water range; model-interaction cursors fix request identities but
allow their lifecycle state to advance. See [Runtime history](runtime-history.md).

## Unreleased source and wire migration

This is a breaking pre-release change with no compatibility aliases:

- Replace observe callbacks with async iteration over watch, and wait_observed
  with wait(on_event=...). Replace TaskGraphWaitResult with WaitResult
- Wait/run/plan observation content uses include_event_content; graph wait's
  include_content selects only its result projection. Watch keeps include_content
- Trace no longer accepts include_content; read bodies using history or
  model_interactions with the returned execution/run locators
- TaskGraphRun.inspect is consolidated into state(include_content=True), which
  also returns node states and the graph event sequence
- Old result.output/status access on SDK wait/run/plan becomes
  outcome.result.output/status. Graph display status is result.wait_status
- RuntimeHistory.task_events/list_events become
  list_task_events/list_execution_events
- CapabilityLoadContext.verify becomes verify_source_revision
- AssetStore and StorageOverlay no longer expose layer-ambiguous numeric history.
  Capture layer-qualified references with AssetStore.resolve_versions(keys) and
  read them with read_versions(refs); numeric history remains backend-local
- Local Skill resources use DirectoryAssetBackend and CapabilityGroup.capture(),
  then AssetSkillSource with the captured reader and SkillSourceRef
- EvaluationRun.report becomes create_report; RuntimeEvaluations.compare becomes
  create_comparison_report. These methods create and persist reports
- Skill/Subagent instructions aliases are removed; use get_instructions.
  EventRepository.append is removed; use append_expected
- ObservationError, OBSERVER_FAILED and OBSERVATION_FAILED replace the old
  task-prefixed observation names. Public watch uses opaque cursors only
- StepEvent.kind/EventKind become event_type/StepEventType with uppercase values:
  AGENT_RUN_STARTED/SUCCEEDED/INTERRUPTED/FAILED,
  MODEL_REQUEST_STARTED/SUCCEEDED/FAILED/CANCELLED, and
  TOOL_CALL_STARTED/SUCCEEDED/FAILED
- LiveDelta/DurableBoundary and ExecutionDelta use event_type. Tool payloads use
  error_code. MODEL_REQUEST_FINISHED and TOOL_CALL_FINISHED keep their distinct
  status-bearing boundary meaning, including failure/cancellation
- Cancellation history uses the explicit current cancellation fact, with no
  legacy failed-plus-cancel-code reinterpretation

StepEvent fields, marker values, archive entries and idempotency projections
change durable wire bytes. Existing development databases, archives, snapshots
and files using the former wire may no longer load. Current writers/readers and
owned test fixtures are updated together without a format/schema version bump.
No startup conversion, dual read, data deletion or real-data migration is run.
Preserve existing data; an offline conversion or deliberate rebuild requires an
explicit separately authorized operation before using it with the new code.

Earlier source cleanup in this branch also removed get_at_version and atomic
writer aliases, renamed EvaluationRecord.experiment_id and
OperationLedgerRepositoryImpl, and centralized model_binding_error. Storage
key ownership changes themselves retain their prior key bytes.
