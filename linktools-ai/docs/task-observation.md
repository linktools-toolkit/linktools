# Observing and waiting for TaskGraphs

`TaskGraphRun.wait_observed()` owns the local observation/wait lifecycle. It
returns one authoritative graph read and observation diagnostics; it does not
cancel, resume, recover, or retry the graph.

```python
from linktools.ai.runtime import TaskGraphRunEvent

async def publish(event: TaskGraphRunEvent) -> None:
    await projection.apply_idempotently(event)

outcome = await run.wait_observed(
    publish,
    cursor=saved_cursor,
    include_content=False,
    timeout_seconds=120.0,
    close_timeout_seconds=5.0,
)
await projection.refresh_graph(outcome.graph, status=outcome.status)
await projection.save_cursor(outcome.cursor)
if outcome.observation_error is not None:
    await projection.show_observation_gap()
```

The projection methods above are application-owned integration points, not
library APIs. This repository does not contain `RuntimeTaskGraphExecutor` or its
UI/storage implementation; downstream integration and end-to-end acceptance
remain separate work.

## State and delivery

- `outcome.graph` is the exact stopping read from the TaskGraph service, projected
  as `TaskGraphInfo` by default. `include_content=True` returns `TaskGraphState`
  with node inputs and also enables observation content. It does not load every
  node output
- `outcome.status` applies the same public status projection as `run.wait()`.
  Raw `graph.status` can be `RUNNING` while public status is `WAITING` when no
  runnable node remains and an unfinished node is waiting
- Terminal statuses (`SUCCEEDED`, `FAILED`, `CANCELLED`, `BLOCKED`), stable
  `WAITING`, and `RECOVERY_REQUIRED` stop the wait. Returning does not imply
  business success. Other callers can change the graph after the returned read
- Dynamic nodes and their states come from that same complete read. The graph
  sequence does not replace the observation cursor and does not define an atomic
  snapshot across graph and execution-history stores
- Callbacks are serial. Only successful callback return acknowledges its cursor;
  interrupted delivery may replay. Coordinate saved cursors with an idempotent
  projection. External effects are not exactly-once
- Stream EOF does not complete the wait. Conversely, authoritative completion
  can stop observation before every event has been delivered. Refresh final UI
  from the returned graph and paged results; use the returned cursor to replay
  remaining observations if needed

Live execution trees cover the root and direct child agents only. Recursive
history is a separate capability. Model request identity remains execution ID,
agent run sequence, and request sequence. Output-repair retry indexes are not
HTTP/SDK transport attempts; no transport-attempt ledger is added here.

## Errors and cleanup

Only proven optional presentation/broker failures become a returned
`observation_error`. State/history reads, authorization, durable decoding,
integrity, lineage and cursor failures propagate unchanged. Callback failure
raises `TaskObservationError(origin="callback")` with its original cause and
last acknowledged cursor. Callback and authoritative observer failures start
bounded cleanup before the stream finishes closing; a stalled close cannot hide
an already-raised primary error. `CancelledError` propagates unchanged.

Errors visible in the same completion/cleanup window have stable priority:
caller or callback cancellation, authoritative failure, callback failure,
wait timeout, then cleanup failure. A successful wait cannot hide an already
completed authoritative or callback failure.

Time budgets must be finite nonnegative numbers (`close_timeout_seconds` must
be positive). Booleans, NaN and infinity are invalid. The composite owns its
monotonic wait deadline and uses a separate finite cleanup budget. Timeout raises
`AIError(TASK_WAIT_TIMEOUT)` with the graph ID and never implicitly cancels the
graph.

Cleanup first closes callback delivery, then requests local cancellation once.
The Runtime retains unfinished observation/wait tasks, including their nested
stream cleanup. A cleanup budget expiry is not successful completion: without a
higher-priority error it raises `TaskObservationError` with `phase="cleanup"`
and `cleanup_pending=True`. An already-entered cancellation-resistant callback
may continue; no new callbacks start. Late task errors are consumed by their
owner. `Runtime.close()` stops these sessions before closing storage, fails
without claiming closure if they remain, and can be retried after they finish.
New observed waits are rejected once Runtime closing starts.

These bounds require a schedulable event loop. Python cannot safely interrupt a
synchronous blocking callback or forcibly terminate a coroutine that ignores
cancellation. Do not put blocking work in callbacks; no thread or process is
created automatically.

## Explicit cancellation and unknown outcomes

Cancellation is application policy. Store one stable idempotency key for one
business cancellation intent and reuse it for every retry:

```python
await run.cancel(idempotency_key=saved_cancel_key, force=False)
```

The control operation can commit after its caller times out or is cancelled.
A bounded `run.state()` readback must establish the actual outcome:

- `CANCELLED`: cancellation confirmed
- `SUCCEEDED`, `FAILED`, or `BLOCKED`: that terminal outcome is confirmed;
  do not rename it cancellation success
- Other status: cancellation is not yet reconciled; retain the key and retry
  the same intent according to application policy
- Readback timeout/failure: outcome unknown; retain graph ID, key and cursor

An application that stops waiting for a pending cancel/readback task must retain
it in its existing task owner and consume its eventual result. Do not discard a
bare `create_task`, infer non-commit from timeout, or automatically enable force.

## Results and paging

`run.results()` remains the output API. Pages bind a graph sequence. A change
within a page reports `STORAGE_CONFLICT`; a changed sequence between pages
reports `CURSOR_INVALID`. Discard the entire unpublished batch and restart from
page one on either conflict, within an application-owned finite retry budget.
Never splice old and new pages. Large batches should use application-owned
bounded staging or a paged cache rather than unlimited memory.

Publish a complete staged batch only after `next_cursor is None`. The result
batch and an earlier observed-wait graph are not a cross-store transaction.
Use `content_included` to distinguish an omitted output from valid JSON null;
respect `max_content_bytes` and fetch large content through `result_ref` as needed.

## Source migration for the unreleased API

There is one TaskGraph service wait operation: `TaskGraphQueryService.wait()`
now returns `TaskGraphState` from its stopping read. Custom implementations and
fakes must do the same. Do not implement it as `wait()` followed by a second
`state()` read. There is no additional `wait_state()` alias. Service `run()` and
Runtime `run.wait()` still expose their own `TaskGraphResult` boundary projection.
Direct service consumers replace `result.node_results` with `state.node_states`.

Other source migrations in this change:

- `storage.atomic_write_bytes/json` are removed. Use `write_bytes_atomic(Path,
  bytes, fsync=False)` / `write_json_atomic(Path, dict, fsync=False)`. Canonical
  JSON sorting, value validation and bytes are unchanged. Former callers needing
  a non-object JSON root or insertion-order bytes must explicitly serialize their
  required bytes and use the byte writer
- Replace `get_at_version(key, integer)` with `get_at_revision(key,
  StorageEntryRevision(integer))`; prefer reusing the returned `entry_revision`.
  Revisions retain their integer representation and overlays keep their existing
  ordered layer-selection semantics. Asset versions and `list_versions` retain
  their separate resource-version meaning
- `EvaluationRecord.experiment_id` replaces its derived `evaluation_id` property.
  Tombstone and cleanup wire fields named `evaluation_id` are unchanged
- The concrete ledger implementation is `OperationLedgerRepositoryImpl`; the
  public role Protocol remains `OperationLedgerRepository`
- Runtime key construction is Runtime-owned. It is no longer exported by generic
  storage; physical `v1/runtime/...` bytes are unchanged
- Generic overlay unknown-layer failure is `STORAGE_LAYER_UNKNOWN`; the Asset
  facade retains `ASSET_VERSION_LAYER_UNKNOWN`
- Provider classification belongs to `model.model_binding_error`. Runtime keeps
  execution/timeout/output semantics; callers must not equate provider retry
  hints with automatic execution retry

No database schema, durable format version, named behavior revision, or key-byte
migration is introduced.
