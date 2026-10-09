# Reading execution content while a run is active

Observation events identify progress; the corresponding Runtime domain owns
its recorded content. Live progress can arrive before its history is persisted.
Retained execution routes coalesce observation writes with one Runtime-owned
timer. The first queued observation starts a one-second window; the shared
tick flushes queued runs. A run reaching 16 observation boundaries flushes
earlier. Each run keeps its own transaction and completion barrier, so a blocked
run does not stop another run's publication. An idle Runtime performs no flush.
Storage latency and contention can extend publication time. A blocked provider
or tool does not need to produce another event for that scheduled flush to run.
Received response/result content is eligible for publication when an individual
model request or tool call finishes. A streamed response stays in memory while that request is still
running, including already-ended response parts. Individual completed tool
results may be published while sibling tools are still running; their pending
transcript group contains complete results, not half-received tool bodies.
While a retained model request is `RUNNING`, its public interaction contains
identity/status metadata and empty request/response content. The prepared input
stays with the active request; required recovery captures remain unchanged.
Completion or a controlled abnormal end publishes the actual prepared input and
any received response. Requesting content early does not force a write or provider completion.
Complete outgoing raw input may already be captured before provider completion;
the deferred body rule excludes half-received response fragments. Model metadata
may be refined during preparation without changing request identity or start time.

With shared retained backing storage, independently opened
`RuntimeHistory.open(...)` readers see the committed prefix while the execution
is active. An immediate read after a live notification can be empty; a fresh
read after publication includes the new facts. Recovery checkpoints remain
durable independently of public observation publication. Terminal handoffs and
Runtime close drain queued observations before completing. Separate in-memory
storage instances do not share state. An explicit TRANSIENT execution route
exposes active content immediately in its owning Runtime/process and discards
it according to its handoff policy.

| Observed fact | Identity | Read operation |
| --- | --- | --- |
| Model request/response | execution ID, agent run sequence, request sequence | `runtime.executions.model_interactions(...)` |
| Function tool call/result | execution ID, agent run sequence, call ID | `runtime.executions.history(...)` |
| Completed assistant text/thinking part | execution ID, agent run sequence, message sequence, part index | `runtime.executions.history(...)` |
| User/assistant transcript | execution ID; current agent run | `runtime.executions.transcript(...)` |
| Execution step facts | execution ID and agent run | `runtime.executions.trace(...)` |
| Request attachment inclusion | execution ID, agent run sequence, request sequence, attachment ID | `runtime.history.attachment_facts(...)` |
| Task result | graph ID, task ID, result reference | the task run's `result`, `result_ref`, or `results` operation |

For a root execution, model interactions include recursive descendants;
selecting a `SUBAGENT` reads only that selected execution, not its subtree.
Ordinary history and trace include a root and its direct children, or only the
selected subagent. Transcript reads the selected execution/current run.
Execution watch traverses the selected execution and its descendants, so its
scope can be broader than a history query. Read a deeper subagent using that event's own
`execution_id`, rather than substituting its parent's identity. Task and
evaluation results remain in their own domains.

Trace always returns step/status metadata and content locators. It has no
include_content option; use those locators with history or model_interactions
when the application needs a body. Keep watch as notification and request
content explicitly from the owning Runtime read API.

## Read one tool call

The default watch keeps safe locator and status metadata, including
`agent_run_seq` and `call_id`, without exposing arguments or results.

```python
payload = event.event.payload  # ExecutionTreeEvent from execution.watch()
page = await runtime.executions.history(
    event.execution_id,
    principal=principal,
    agent_run_seq=payload["agent_run_seq"],
    tool_call_id=payload["call_id"],
    include_content=True,
)
```

The result has the same `ExecutionHistoryItem` shapes as unfiltered history:
`tool_call` contains the model's arguments, `tool_result` contains the returned
content, and `retry` contains the retry prompt. These are not effective
post-validation parameters or a separate external-effect ledger. A call can be
visible before any result exists. A failed or cancelled call without a returned
part does not acquire a fabricated result.

An empty page means no matching committed fact is currently available. During
an active run it does not prove that a live-observed request or call is absent.
Keep its exact execution/run/request or execution/run/call identity and show a
pending detail until a later refresh supplies the content. Do not infer the
identity from display order, a timestamp, or `step_index` alone. Use the existing
watch/wait flow for progress and refresh the existing detail read; there is no
separate body-polling API.

A result item with
`content_included=True` and `content=None` represents a real JSON null.
`content_included=False` means the body was omitted, either because it was not
requested or because the response content budget was exhausted. Requesting
`include_content=True` does not guarantee that every body fits. Tool status
and external-effect uncertainty remain separate from whether a result part
exists; `EFFECT_UNKNOWN` is not success.

## Read one assistant part

`ASSISTANT_PART_COMPLETED` includes the original response-part index and its
one-based history message sequence:

```python
page = await runtime.executions.history(
    event.execution_id,
    principal=principal,
    agent_run_seq=payload["agent_run_seq"],
    message_seq=payload["message_seq"],
    part_index=payload["part_index"],
    include_content=True,
)
```

`agent_run_seq` limits the query to that exact execution and run. All selectors are optional and combine with AND; they may also be used
independently. Different runs and child executions may reuse a call ID. Selectors,
execution identity, content mode, and tenant are bound to the cursor.

When present, `part_index` is the zero-based coordinate in the raw message's
`parts`, not its position in the projected history page. Message-level
`instructions` are synthetic, have `part_index=None`, and do not shift system,
user, or response part coordinates. Tool results and tool retries retain their
call-based stable identity across pending-tail completion and archival; select
them with `tool_call_id` rather than assuming a positional locator.

A recovered request may replay a previously recorded tool call. Its raw model
response remains intact in `model_interactions`; `history` exposes the original
call and result once, while retaining new response parts at their original
coordinates. A pending replay is checked against the original call before it
is omitted. Reusing an ID with different arguments is an integrity conflict.

An assistant-part notification can precede completion of the containing model
request, so its content may remain unavailable until that request finishes.
When the owning Runtime handles request failure, cancellation, or shutdown,
it retains received partial response content with an interrupted state and the
actual terminal request status. An uncatchable process termination cannot flush in-memory
content. Streaming text deltas remain live updates. `transcript` contains user and assistant text; thinking,
tool arguments and tool returns belong to `history`.

A failed or cancelled model interaction can contain received partial response
content. Its status and error remain authoritative; a response body does not
imply success or a completed model output. `INTERRUPTED` identifies an abandoned
running record recovered without an observed completion, so it has no invented
response, completion time, or usage.

## Request identity and step filters

A model request is identified by `(execution_id, agent_run_seq,
model_request_seq)`, exactly the identity returned by `ModelInteractionItem`.
`agent_run_seq` is a one-based run sequence within an execution, not a
session turn or child-execution ID. `model_request_seq` is a one-based logical
model-request sequence within that run. Its journal includes agent requests,
output-correction retries and compaction requests; it is not a provider request
ID or a count of transport attempts. The sequence continues from recorded
requests when the same run is recovered.

`step_index` retains the SDK's execution-step meaning within the run; it is
non-negative but is not guaranteed to start at zero. Several requests can
belong to the same step. It is useful for grouping, but cannot replace the
request identity. `message_seq` is the one-based raw transcript message
coordinate on `ExecutionHistoryItem`, `TranscriptItem`, and `SessionHistoryItem`.
It counts messages, so multiple projected parts from one message share the same
value. `part_index` is the original zero-based part coordinate inside that
message, even when other parts are filtered out.

`ExecutionTraceItem.step_event_seq` is the raw `StepEvent` ordinal plus one,
within its agent run. Trace filtering does not renumber it, so results can have
gaps. It is distinct from `step_index` and from the one-based durable
`ExecutionEvent.event_seq` / `TaskEvent.event_seq` owned by their execution or
graph. `ExecutionStreamEvent.durable_seq` is that durable event coordinate, or
`None` while the live event has no durable coordinate, including live deltas
and uncommitted semantic progress. Such events do not advance the signed durable
watch cursor. Reconnecting from that cursor can redeliver their later persisted
form; reconcile by exact logical identity and lifecycle state. A non-null event
coordinate certifies that event's persistence, not a newer transcript or model
interaction read boundary. None of these coordinates are interchangeable.

Session timeline references use one-based `turn_seq` within their owning
session; `timeline_parent_turn_seq` fixes the parent-session turn cutoff for
a fork. These do not replace `SessionTurnItem.ordinal`. Generic storage fact
and operation sequences remain separate storage coordinates.

These names define the current pre-release API and wire contract. The field
rename is breaking: callers and current serialized data must use the new
names; compatibility aliases are not provided.

For assistant response parts and tool calls, history's `model_request_seq`
identifies the request that **produced** the response. For tool returns and
call-specific retry prompts it identifies the request that **initiated the
call**, even when a later model request carries that return. History's
`step_index` is the originating model request's step. The existing successful
model step event records the actual raw response's `message_seq`; trace
exposes that coordinate on its model response item. Tool trace events retain
their own occurrence step and the originating request sequence.

User/system messages, unassociated retry prompts, incomplete response parts,
and facts without a recorded association have `model_request_seq=None` and
`step_index=None`. No association is guessed from time, message order, or
matching text. Compaction requests appear in model interactions and trace but
do not invent a response in conversation history. To inspect all inputs carried
by a request, read its `ModelInteractionItem.request`; history is an occurrence
view, not a duplicate of every model input context.

Both `history(...)` and `trace(...)` accept `agent_run_seq`,
`model_request_seq`, `step_index`, and `tool_call_id`. History additionally
accepts `message_seq` and `part_index`. Without a run selector, a request
or call selector matches each run in the normal query scope; always retain
returned execution/run coordinates to disambiguate results. For example:

```python
page = await execution.history(
    agent_run_seq=1, model_request_seq=2, include_content=True, limit=20,
)
trace_page = await runtime.history.trace(
    execution.execution_id, principal=principal,
    agent_run_seq=1, model_request_seq=2, limit=20,
)
# Continue with the same selectors and content mode.
page = await execution.history(
    agent_run_seq=1, model_request_seq=2, include_content=True,
    limit=20, cursor=page.next_cursor,
)
```

Results remain `Page` objects with signed cursors. An unknown valid selector
returns an empty page. Invalid selector values are rejected; changing or
removing a selector while continuing a cursor is rejected. Pagination freezes
association facts at its event high-water mark: a partial response with a null
association does not gain one mid-page after the request succeeds. A fresh
query can observe the completed association.

## Visibility, paging and persistence

For retained execution routes, public history reads the canonical EXECUTION
owner. Admission, prepared model requests, completed response parts, and
model-facing tool returns enter the next observation batch; raw streaming
deltas do not become complete history parts. Checkpoint and terminal operations
force pending observations through their required commit boundary. A public
read captures the committed run identity and transcript, event, and model-request
high-water marks together. Immutable body references can then be resolved
outside that read boundary. Readers do not fall back to producer-local staging
when a retained fact has not yet been published.

Completed messages retain their normal transcript coordinates. An in-progress
tail freezes its visible part identities when a page is started: parallel tool
completion order can differ from final SDK message order. Continuing that cursor
after more results arrive, finalization, or reopening reads the original set
exactly once. Start a fresh query to see newer facts. Pending tool results retain
call-based identities and `part_index=None`.

Independent readers of shared retained storage refresh durable state rather
than relying on the producer's local staging, a live Runtime registry, or
recovery checkpoint bodies. The explicit TRANSIENT route reads its existing
process-local recorder owner through the same read contract; it creates no
retained archive and cannot provide cross-process readback. A dropped
notification does not drop the committed fact. Completed retained publication
wakes subscriptions; observation uses durable catch-up and rereads
previously running identities. Process-local subscription generations are wake
hints rather than durable cursors. A sudden process failure can lose recent
uncommitted observations, including prepared inputs or completed response parts
from a live-observed request. Only the committed prefix survives, subject to the selected storage
and retention policy; recovery must not infer a provider outcome from missing
history.

Bodies are opt-in in content-bearing read responses; trace remains metadata-only. Model interaction and attachment-metadata reads
do not resolve unrequested bodies. Existing archived history/transcript
projection may still decode message chunks to identify items even when
`include_content=False`; it does not return those bodies to the caller.


Model interaction cursors freeze request identities and high-water marks, not
request lifecycle state. A captured RUNNING request can be terminal when a later
page reads it. Refresh without a cursor to discover newly started requests.
Model interaction and usage reads include committed active identities.
`UsageSummary.running_requests` and `interrupted_requests` distinguish pending
or interrupted outcomes from succeeded, failed, and cancelled requests. Unknown
usage and duration remain explicitly counted; missing per-request values are
`None`. Recovery records an interrupted unknown provider outcome without
inventing a successful or cancelled response.

## Known-execution metadata reads

Graph observation already performs metadata compensation internally; ordinary
graph consumers should use its single watch/wait callback. Lower-level readers
that already know an execution can use three narrow `runtime.history` operations
without recursively rediscovering its descendants:

```python
subscription = await runtime.history.subscribe_model_interactions(
    execution_id, principal=principal,
)
generation = 0 if subscription is None else subscription.generation
try:
    boundary = await runtime.history.capture_model_interaction_cutoffs(
        execution_id, principal=principal,
    )
    for cutoff in boundary.cutoffs:
        after = 0
        while after < cutoff.model_request_seq:
            items = await runtime.history.read_model_interaction_metadata(
                execution_id,
                principal=principal,
                agent_run_seq=cutoff.agent_run_seq,
                after_model_request_seq=after,
                through_model_request_seq=cutoff.model_request_seq,
                limit=200,
            )
            await consume_metadata(items)
            after = items[-1].model_request_seq

    # A generation observed before capture retains racing changes.
    if subscription is not None:
        generation = await subscription.wait(generation)
        # Read new cutoffs and reread previously RUNNING identities as needed.
finally:
    if subscription is not None:
        await subscription.close()
```

Always close a subscription when its owner finishes, including failure or
cancellation while reading. Its generation is a coalesced process-local wake
coordinate, not a persisted event sequence or resumable cursor. A standalone
`RuntimeHistory.open(...)` reader returns None for the subscription. All three
operations authorize the selected execution and tenant.

`ModelInteractionReadBoundary.cutoffs` bounds admitted identities per Agent run.
For the canonical execution owner, `durable_cutoffs` contains the same committed
identities, including running requests. `durable_history_available` requires
complete coverage of the declared runs, including active runs; it does not claim
that volatile storage survives a restart. `local_staging_available` is false for
these canonical reads and does not determine active-run coverage. For an
admitted TRANSIENT run it is true while the process-local owner remains
available; `durable_history_available` stays false. Disabled or
unavailable history is explicitly marked; authorization and integrity failures
are not converted into empty successful reads.

Metadata reads return a bounded tuple of contiguous request identities, always
with `content_included=False`, an empty request and no response. They do not
resolve content payloads. Depth is relative to the selected execution (zero);
the graph observer supplies its known tree-relative depth. To reread one active
request at sequence N, use `after_model_request_seq=N - 1` and
`through_model_request_seq=N`. Reusing an old history page cursor cannot detect
a completion for an identity already passed by that cursor.

A fixed request cutoff does not freeze lifecycle status. Each logical request
has one canonical lifecycle record, so a captured running identity can resolve
to its later terminal state. Missing identities inside a declared range,
inconsistent ownership, and conflicting terminal metadata raise integrity errors
rather than fabricating coverage. An uncommitted start is not publicly visible.

## Execution ownership and recovery

SQL and SQLite permit several Runtime instances to use one namespace. Opening
another Runtime does not take over an admitted or running execution, or finish
an unhanded-off cancellation. Separate sessions can run concurrently; one
session's active-execution admission still uses its existing optimistic claim.
Runtime does not add an ownership heartbeat or a namespace-wide database lock.

After an executor crashes, confirm it has stopped, then call the existing
`runtime.executions.recover(execution_id, principal=...)`. Recovery competes for
the revision authorized by that invocation, preserves producer and tool-effect
fences, and rejects a worker still running in the same Runtime. Unknown tool
effects retain the existing `RECOVERY_REQUIRED` resolution flow. Each fresh
call is a deliberate takeover, not a retry-idempotent request. If the response
is uncertain, inspect canonical state before deciding whether another recovery
is appropriate. A prepared start with no recovery checkpoint must use its
original start/idempotency flow rather than inventing a checkpoint.

Filesystem execution storage retains automatic crash recovery because its
existing writer lock supplies exclusive ownership. SQL startup can still finish
durably handed-off work, including deferred continuations, prepared terminal
handoffs, and terminal cleanup. A plain `FINALIZING` status alone is not proof
that an active producer relinquished ownership.

Remote cancellation reports `CANCELLING` and `cancelled=False` until the owner
reaches an existing model-request or individual-tool completion boundary. The
owner records received content and completed effects, stops further dispatch,
and drains terminal history before sealing cancellation. Local cancellation
still interrupts its local worker immediately. A blocked provider can remain
`CANCELLING` until its boundary or configured timeout; no cancellation poller or
finite completion-time guarantee is added. After a crash, explicit recovery
can retain only content that was committed before the crash.

A producer that loses a claim stops locally without changing the canonical
execution to failed or cancelled. Waiters and streams continue from durable
state owned by the winning producer. Live events with `durable_seq=None` remain
provisional: previously delivered deltas cannot be retracted, and replay after
handoff does not promise exactly-once delivery of those provisional events.

## Attachment paging

Attachment cursors likewise retain request high-water marks, while continuing
after a fact identity ordered by agent run, request, fact kind, attachment ID,
and occurrence ordinal. Initial input acceptance sorts before request facts.
Acceptance and inclusion are distinct facts; repeated inclusions of one
attachment retain their multiplicity, numbered within that fact/attachment pair.
Removing an indistinguishable repeated inclusion contracts that ordinal range.
Preparation may update positions or remove inclusion without shifting unrelated
identities, and accepted occurrences remain visible after preparation and archive.
Continuation reads current values, including newly added facts after its last
identity; facts added before it require a fresh query. New requests above the
captured high-water marks remain excluded. A cursor contains one identity and
one cutoff per captured agent run, never attachment bodies or a fact snapshot.

## Model context compaction

Runtime derives a best-effort compaction target from the current SDK model's
`context_window`, including its profile override and SDK registry lookup. It
subtracts an explicitly configured `max_tokens`, merging request settings over
model defaults. An explicit smaller context target still wins. Unknown windows
remain unknown; without an explicit target, Runtime only deduplicates eligible
file reads. It does not invent a model table or an output-token allowance when
the provider's default is unknown.

The trigger uses Harness's provider-usage anchor and text estimator. The current
history, system/instruction text, visible function/output tool schemas and
structured output schema provide an additional estimate floor, without charging
those schemas again on top of a usage anchor that already covers them. Standard
SDK dynamic instructions and tool preparation run before this projection. Raw
history remains unchanged; model interactions record the prepared provider view.

These are compaction estimates, not exact token limits or a guarantee that every
request fits. Binary inputs retain their separate byte/count limits; their token
cost, native-tool/provider serialization overhead and unknown output defaults
remain provider-specific. Irreducible prompts and preserved recent messages may
still exceed the target, and the summary request itself must fit the selected
model. Custom capabilities that enlarge input after the compaction hook remain
responsible for that late change. Runtime neither rejects a request solely from
this heuristic nor adds a token-counting network request on every model call.
