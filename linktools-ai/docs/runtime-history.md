# Reading execution content while a run is active

Observation events identify progress; the corresponding Runtime domain owns
its recorded content. Live progress can arrive before its history is persisted.
Detailed execution persistence requires `TraceCapability()` registered
at Runtime scope; storage retention alone does not select it. Without this
capability, the owning Runtime exposes its existing live staging content while
retained locally. Final results follow execution retention. Session management is supplied by the default `SessionCapability`; retained
conversation content and model context for subsequent turns require
`ExecutionCapability(retain_history=True)` and follow conversation retention. An unretained
completed turn makes subsequent resume/fork fail with
`SESSION_HISTORY_UNAVAILABLE`; it cannot be repaired by enabling history retention
later. See
[optional execution persistence](runtime-capabilities.md#optional-execution-persistence).

For runs with restart checkpoints disabled, model-boundary control checks may
reuse one execution/history-head observation captured within the last three
seconds. Both records share the same source capture and expiry. Without a hot
record cache, each check reads the source. Tool admission, recovery checkpoints,
terminal commits and result consumption retain their authoritative transaction
checks. Task-owned runs and their delegated descendants keep source-only model control
checks, including on transient execution routes.

With durable trace enabled and an explicit `CommitPolicy.batch(buffer=...)`,
Trace coalesces observation writes through its owned cache-buffer scopes. The
default batch policy flushes at 128 observation boundaries or after three seconds.
In batch mode without a write buffer, observation writes are synchronous. Each run keeps its own transaction and completion barrier, so a blocked
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
stays with the active request. Separately enabled restart checkpoints preserve
recovery context without publishing public trace.
Completion or a controlled abnormal end makes the actual prepared input and
received response eligible for the selected Trace publication boundary. Requesting content early does not force a write or provider completion.
With `TraceCapability(record_http_bodies=True)`, complete outgoing raw HTTP
input may be captured before provider completion; the deferred body rule excludes
half-received response fragments. Requests containing `ReferencedContent` instead
omit both raw bodies with an explicit lifecycle reason; their prepared input keeps
the immutable descriptor. See [recoverable reference inputs](file-tools.md#recoverable-reference-inputs).
Raw HTTP body capture is disabled by default. Model metadata
may be refined during preparation without changing request identity or start time.

`TraceCapability(commit=CommitPolicy.terminal(buffer=write_buffer))` freezes the
same eligible projection and dependency bytes but does not publish on a batch
timer. The verified prefix joins the existing terminal source transaction;
required safety or explicit publication boundaries remain independent. Without
a buffer, terminal Trace keeps its existing local staging. Working reads still
come from that owner, while independent History reads only source-confirmed
facts. A shared buffer is not an independently readable History archive or a
restart/takeover mechanism. See [commit boundaries](runtime-capabilities.md#independent-commit-boundaries).

With shared retained backing storage, independently opened
`History.open(persistence=p)` readers see the committed prefix while the execution
is active. An immediate read after a live notification can be empty; a fresh
read after publication includes the new facts. Opted-in recovery checkpoints
remain durable independently of public observation publication. Terminal handoffs resolve their selected observations with canonical control.
Runtime close drains eligible queued observations and reports unresolved
source/control or acknowledgement cleanup; it cannot make an unresolved terminal
prefix durable by silently changing its publication policy. Separate in-memory
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

## Read one model request

Use the exact identity returned by a model interaction or notification to read
that request's original client bodies:

```python
page = await runtime.history.model_interactions(
    item.execution_id,
    principal=principal,
    agent_run_seq=item.agent_run_seq,
    model_request_seq=item.model_request_seq,
    include_client_requests=True,
)
```

`model_interactions(...)` accepts the optional positive integer selectors
`agent_run_seq` and `model_request_seq`. They combine with AND. As with history and trace, `agent_run_seq` selects only that exact
execution and run; with both sequences the result identifies at most one
logical request. With only `model_request_seq`, the selector matches each run
in the normal recursive scope. Without selectors, the existing scope is
unchanged. Read child requests using the child's own `execution_id` so that
its execution authorization applies.

`include_client_requests=True` loads original client attempt bodies independently
of `include_content=True`, which loads the SDK request/response projection.
Both default to false on the public API. Selectors apply before either body
projection is resolved. An unknown valid identity returns an empty page. Keep
the same selectors and content flags when continuing a cursor; adding,
changing, or removing them is rejected. `cutoffs` remain fixed upper bounds,
not lower bounds or substitutes for exact request selectors.

Each retained `ModelClientRequest` also exposes optional `diagnostics`, independently
of HTTP body capture. It belongs to that exact dispatch, not another request or
the execution as a whole. `started_at` is UTC; `*_ns` observations are monotonic
elapsed nanoseconds from that attempt's start. The mapping can include header,
first/last chunk and close timings, received byte/chunk counts, HTTP version and a
sanitized protocol request ID, transport read/close errors, and separate `sdk_*`
stream entry, exhaustion, cancellation and cleanup observations. Error observations
contain qualified exception types, not exception text or headers. Missing keys
mean unobserved; buffered responses do not invent stream chunk timings. See
[protocol terminal semantics](model-protocols.md) before interpreting a response's
`complete` flag. The mapping is retained in metadata-only archives too.

## Read complete failure diagnostics

Read the failed execution's authoritative `result()` to obtain its
`error_diagnostics`; lightweight execution inspection remains a summary. Cancelled
execution results retain their cancellation status/code, not a fabricated failure
diagnostic; observed cancellation timing belongs to the affected client attempt.
The public
`ErrorDiagnostics.to_payload()` returns every collected diagnostic field. The
original `exception_type`, bounded `exception_message` and `cause_digest` remain
summary fields. New failures also collect an optional `exception_chain` (outermost
first, with `root`, `cause` or visible `context` relations and full messages) and
`traceback` (all frames and visible exception-group branches, without locals).
Python traceback display limits do not truncate the durable diagnostic. Normal
exception context suppression and cycle detection still apply.

Durable output-schema validation has a stricter existing privacy contract: rejected
input and schema literals must not be retained. Its diagnostic preserves original
chained frame locations and types plus the safe validation explanation, but omits
original messages, notes and source lines. The traceback states those omissions;
`exception_chain=None` does not claim a fully collected message chain.

Provider failures for recoverable reference inputs also omit arbitrary exception
messages, notes and source lines, since an SDK or server can echo the temporary
file body. Their optional `content_omission_reason` identifies that policy.
Exception types, relationships and frame locations remain available without locals.
This does not weaken ordinary credential-redacted diagnostics or fabricate absent
content for older records.

Collection uses the shared credential redactor before persistence, including the
bounded summary. It does not retain arbitrary exception attributes, connection
headers or local variables. Existing access, retention and deletion rules apply;
complete diagnostics are not automatically safe for publication outside that scope.
Terminal failure events, retained results and reopened history use the same
diagnostic projection. Optional fields absent from older records remain absent or
`None`; enabling capture cannot reconstruct prior chains, stacks or stream timing.
These additive diagnostic fields keep the current durable format version and do not
alter model, capability or execution identity. No automatic database migration is
required.

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
content, and `retry` contains a legacy SDK retry prompt. On SDKs with native
retry feedback, a tool retry remains a `tool_result` with a failed status, not a
successful result; its original `ToolReturnPart` has `outcome="retried"` and
optional `retry` information. Non-tool output feedback has item kind
`retry-feedback`, without a fabricated tool name or call ID. Its content
includes the feedback and retry cause, attempt and maximum retry count.
These are not effective
post-validation parameters or a separate external-effect ledger. A call can be
visible before any result exists. A failed or cancelled call without a returned
part does not acquire a fabricated result.

The durable message codec preserves both the legacy retry prompt and the native
feedback/tool-return shapes. Native retry information is retained through
checkpoints and history, including explicit unknown (`None`) retry budgets.
Ordinary tool returns without retry information keep their existing encoding.
Reading native retry semantics on an older SDK that cannot represent them fails
with `STORAGE_VERSION_UNSUPPORTED`, rather than silently translating or dropping
the feedback. These additive message fields do not change the format version.

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

An interrupted SDK tool-return placeholder remains in the raw transcript and
has `INTERRUPTED` history status. It is not a confirmed result and does not
prevent recovery from recording the eventual result under the same call ID.
An attempt ending with typed `TOOL_EFFECT_UNKNOWN` retains its error event,
shown in trace as `TOOL_ERROR` with `EFFECT_UNKNOWN` status. After effect
resolution, the real success or failure supplies the call's terminal status;
the earlier error and placeholder remain unchanged.

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

Observed streamed model interactions expose `stream_attempts`. `None` means no
attempt scope was observed (including older records); an empty tuple means an
observed scope had no attempts. Each item has a one-based `stream_attempt_seq`,
terminal status, timing, observed usage, and a zero-based half-open
`client_request_start`/`client_request_end` range into the HTTP request array.
An SDK invocation is not an HTTP count: native fallback can dispatch multiple
providers within it. Several successful continuation segments can contribute to
one logical response, so `SUCCEEDED` on an attempt is not a separate final answer.

Failed or cancelled attempt bodies are diagnostic TRACE evidence, not conversation
history. Successful attempts do not duplicate the logical response body. If a
later continuation fails after an earlier successful segment, the logical
interaction may retain the actual interrupted aggregate in TRACE, still marked
failed or cancelled. Checkpoint and execution projections omit these failed
observed-stream bodies, and `include_content=False` omits them from public reads.
Known failed-attempt usage is retained without inventing zero for an unknown value.

Live attempt previews are provisional. Bundled consumers use the committed logical
response to correct affected previews at completion; if retained content is absent
or cannot be read, they mark the final content unavailable rather than treating
concatenated deltas as a verified answer. This does not change execution status or
enable additional trace retention.

Where response content belongs to the canonical transcript, agent response capture
reuses that archive range. This avoids another response body while keeping
metadata-only reads free of body loads. Compaction and responses rejected by tool-call integrity checks
retain independent response evidence. Request settings and attempt metadata
remain available. Original HTTP request/response bodies are recorded only with
`TraceCapability(record_http_bodies=True)`; this diagnostic option can duplicate
full prompts and may contain sensitive content. Disabling it does not remove
canonical model messages, tool evidence or usage facts. An explicit recoverable
reference input overrides raw body capture for its logical request, recording
`diagnostics.body_omission_reason` while preserving attempt metadata. Its canonical
messages keep descriptors rather than file bytes. Archive relocation may materialize
content again; Trace, Checkpoint, and Conversation retain their own lifecycle
roots rather than depending on another capability's retained history.

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

With `ExecutionCapability`, each admitted completed session turn retains its canonical
user input and public final output alongside the transcript range. `SessionTurn.output`
is the committed execution output, including structured JSON or JSON null; it is
not reconstructed from model text parts. Check `status` to distinguish successful
null output from failed or cancelled turns, whose output is absent. Turns with no
model messages still retain their input, status and safe error details.

Completed timeline entries are owned by Conversation and remain readable after
Execution records are removed. Object-backed input and output bytes are retained
in the Conversation object store before publishing the turn. Runtime close and
session close do not delete durable conversation history; existing offline object
cleanup uses the retained turn references as roots. Forks retain the selected
parent turn prefix. Model context compaction does not rewrite the canonical input
or final output. Without `ExecutionCapability`, standalone execution results
retain their existing execution-scoped lifetime.

A same-group terminal commit publishes execution and conversation together. With
separate transaction groups, transcript preparation can precede execution commit,
but the completed turn envelope is published only after terminal truth is durable,
with session release. An active timeline never infers successful output from a
prepared transcript. If execution terminal state is durable but the retained turn
receipt is still unconfirmed, result and retained timeline reads report a typed
storage recovery error. Admission and execution retention remain held until the
existing prepared handoff is reconciled; execution business status is not rewritten.
Malformed retained data remains an integrity error. Failed or cancelled turn
input can appear in the timeline without being added to the next model context.

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

For executions admitted with durable trace and a retained execution route,
public history reads the canonical EXECUTION owner. Admission, prepared model requests, completed response parts, and
model-facing tool returns enter the next observation batch; raw streaming
deltas do not become complete history parts. Terminal operations and Runtime
close drain queued observations. Restart checkpoint commits remain independent
of public observation publication. A public
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

Independent `History.open(persistence=p)` readers refresh persisted facts and
may see validated pending metadata through `p`'s configured shared coordination.
They do not discover another Runtime's local staging, even when both borrow the
same Persistence in the same process. `runtime.history` is the producer-bound
view: it additionally reads that Runtime's transient recorder and can expose
content that has not been retained. These are different visibility scopes, not
interchangeable aliases. Neither reader starts another execution producer.
The explicit TRANSIENT route creates no retained archive and cannot provide
cross-process body readback. Shared coordination metadata does not turn
producer-local request or response bodies into shared retained content. A dropped
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
`History.open(persistence=p)` reader returns None for the subscription. All three
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
are not converted into empty successful reads. For an independent reader with
no retained trace, both availability flags can be false and cutoffs empty while
`runtime.history` reports local staging. Empty pages or `next_cursor=None` alone
mean only that this read has no more visible items, not that the execution's
history is complete. Consult `ModelInteractionReadBoundary` for coverage;
unavailable completed content raises `EXECUTION_HISTORY_UNAVAILABLE` along the
existing content-read contract. Task projection coverage likewise reports
unavailable executions separately from observed local or durable content.

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

For bounded live refreshes, `runtime.history.read_model_interaction_updates()`
returns `(boundary, items)`. Optional `after_cutoffs` contains per-run known
prefixes as `UsageReadCutoff` values; omitted runs start at zero. The operation
captures current admitted identities and reads at most `limit` metadata items
across all runs (default 200). The budget is allocated to run suffixes in boundary
order; returned ranges may be short, including omission of later runs. An
unchanged suffix returns an empty tuple with its current boundary. To finish a larger captured range, use
`read_model_interaction_metadata()` with the returned boundary's fixed ceilings;
calling the update operation again captures a new boundary.

The built-in update operation authorizes once before reading its fresh boundary
and first metadata page in one short metadata-owner scope. Authorization runs
outside that scope. Later pages authorize independently and may observe newer
lifecycle states without expanding their captured identities. Custom services
may optionally implement this public operation; otherwise `History`
composes their existing capture and metadata operations, preserving their
authorization and overrides. Custom history readers and service subclasses keep
that same existing-operation composition.

Live graph and execution observers use the update operation for new suffixes
and contiguous active tails. Sparse active ranges retain separate bounded reads.
This changes the number of internal authorization callbacks for a live refresh,
not the requirement to authorize each public query. Subscriptions still precede
snapshots. Final compensation still fixes model cutoffs before tree replay and
reads model states afterward; it does not reuse live prefetched rows.

## Execution ownership and recovery

SQL and SQLite permit several Runtime instances to use one namespace. Opening
another Runtime does not take over an admitted or running execution, or finish
an unhanded-off cancellation. Separate sessions can run concurrently; one
session's active-execution admission still uses its existing optimistic claim.
The ordinary durable path does not add an ownership heartbeat or a
namespace-wide database lock. Eligible deferred turns use the optional cache
turn-coordination lease while a turn has no database record; see
[independent commit boundaries](runtime-capabilities.md#independent-commit-boundaries).

With `ExecutionCapability(retain_history=True)` and non-transient conversation storage,
successful standalone executions retain
their actual final model context for
`execution.fork`, including imported history and tool calls and returns, even
without durable trace. This single canonical terminal checkpoint follows the
conversation storage route and remains available after reopening durable storage;
it does not retain the full intermediate trace or enable restart checkpoints.
An execution without the admitted conversation guarantee cannot gain fork support
from trace or checkpoint data: fork reports `CAPABILITY_REQUIRED_MISSING`.

With `ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal())`, eligible
terminal-only source turns retain this context in the existing final
owner batch in one database write transaction. An explicit terminal payload
buffer also freezes the retained final context when Trace is absent. Newly
captured object bodies are materialized during source preparation and still have
their own I/O. Buffer admission does not promise one network operation or recovery
after process loss. Fork and retry executions retain their existing durable-path admission
and do not inherit that one-write-transaction guarantee.

After an executor crashes, confirm it has stopped, then call the existing
`runtime.executions.recover(execution_id, principal=...)`. Recovery competes for
the revision authorized by that invocation, preserves producer and tool-effect
fences, and rejects a worker still running in the same Runtime. Unknown tool
effects retain the existing `RECOVERY_REQUIRED` resolution flow. A prepared
start with no recovery checkpoint must use its original start/idempotency flow
rather than inventing a checkpoint.

Pass `idempotency_key=...` to retry one recovery control operation, including
through `execution.recover(idempotency_key=...)`. Its receipt and new producer
claim commit atomically. Reusing that key never admits another producer;
changing the actor or execution under the key raises `IDEMPOTENCY_CONFLICT`.
A `SUCCEEDED` recovery receipt confirms launch admission, not business success;
a worker that later fails or is cancelled does not change that control result.
A `CANCELLED` receipt means recovery completed cancellation before launch, which
can happen without a new producer.
The admitted execution keeps its original principal. If recovery instead
establishes a failed terminal outcome (for example, a conflicting session), the
keyed call and its retries raise the recorded error; no launch success is
reported. This does not change the unkeyed recovery return behavior.

If admission committed but launch or receipt settlement is uncertain, retrying
the same key raises `STORAGE_RECOVERY_REQUIRED`; it does not relaunch. Inspect
canonical state and confirm the previous executor has stopped before choosing
a new key. Caller cancellation waits for the owned control operation to settle
before propagating cancellation. Omitting the key preserves the existing
non-idempotent takeover behavior.

Recovery receipts are excluded from terminal-operation compaction so an old
key cannot silently become a new takeover. They remain in durable storage; this
API adds no receipt expiry or automatic retention policy.

A fresh `TaskGraphRun.recover()` is an explicit takeover request for the graph's
recoverable bound Agent executions, including running executions on handed-off
`WAITING` nodes. Confirm that their previous executors have stopped before
requesting recovery: this control does not detect remote liveness. The native
backend rejects a worker still running in the same Runtime. Ordinary Runtime
opening, graph start, and graph wait retain their existing behavior and do not
implicitly invoke this graph-recovery control.

The control actor needs `TASK_RUN` on the graph; standalone execution recovery
still needs `EXECUTION_RECOVER`. Graph recovery preserves the admitted execution
principal and verifies durable graph/node/execution bindings before takeover.
Executions outside those bindings are not recovery targets. Child recovery keys
derive from the graph recovery request, so retrying a partly recovered graph
does not take over an already recovered child again. A later recovery boundary
requires a new graph recovery key.

Filesystem execution storage can automatically recover admitted checkpoint-enabled
executions because its existing writer lock supplies exclusive ownership.
Storage alone does not grant restart recovery. SQL startup can still finish
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

## Resource declarations and read scopes

`PersistenceStorage` is a reusable resource declaration. `Persistence.open(...)`
binds repositories, physical routes, retained-content readers, and maintenance
operations once. `Runtime.attach(persistence=p)` and
`History.open(persistence=p)` borrow that owner; closing either view does not
close `p` or borrowed backends. Custom routes and shared coordination are
configured on Persistence, not supplied again as Runtime features to a reader.
Use `Snapshot` for portable snapshots.

Keep `runtime.history` when reads must follow that Runtime's producer-local
staging. Use independent `History` for retained readback without initializing an
Agent Runtime. Both expose the same read types, but availability is scoped to
the selected reader, so a consumer must not promote a partial view to complete
history. `runtime.executions.capture_input(...)` captures Agent-owned execution inputs;
`engine.inputs` is the public capture and replay facade for a `TaskEngine`.

### Dataclass defaults and existing-session compatibility

Generic dataclass decoding follows the model's constructor declaration for both
canonical and persisted wire values. A missing `init=True` field with a declared
`default` or `default_factory` is omitted from constructor arguments, so the
constructor supplies its normal value. Fields without defaults and `init=False`
derived fields remain required. Explicit values still undergo the original type,
constructor and derived-value checks; an invalid explicit value never falls back
to a default. Wire types and schema versions are unchanged. Format-specific
adapters retain their projections and compact encodings; ordinary dataclass
fields use the same declared-default rule. Writers retain their existing field
sets. Reads do not rewrite or migrate source records.

Defaults mean what the model declares, not a recovered historical value. For
example, omitted budget counts use their declared zero, an omitted optional
resource owner uses `None`, and a time-dependent default factory runs at read
construction time. The codec cannot distinguish a genuinely older omission from
a field removed from a current record; preserve original byte-integrity checks
and do not edit payloads to suppress errors. Mutable factories create a fresh
value for each decoded object, and cross-field constructor invariants still apply.

For model interactions, omitted `attachments` means the declared empty tuple,
while omitted `client_requests` means uncaptured (`None`), not an observed empty
attempt list. Client-request omissions use the declared `capture_error=None`,
`response=None` and `record_http_bodies=True`. That last default is missing-value
metadata, not evidence that bodies were historically captured: resolution never
recreates absent body bytes. Explicit `False` remains `False`, and a body reference
with explicit recording disabled is still rejected. Task-node views use the same
default rule for ordinary fields while retaining their existing durable projection:
lease ownership and resolved dependencies remain outside that projection.
Format-specific TaskNode input semantics and explicit dependency/failure policies
retain their custom required-field contracts.

The owned model-message v1 codec follows the same constructor-default rule for
mapped message, part, retry, instruction and usage fields, including the explicit
conversation-ID and attempt-duration wire mappings. Wire `version`, `kind` and
`part_kind` discriminators remain explicit. Durable tool call/result/retry IDs
also remain required: generating independent authoring IDs on restore would break
call/result correlation. An omitted clock factory runs at read time; it is not
proof of a historical timestamp. Exact pending continuation still fails closed
when a missing clock value prevents proving that the recorded part is unchanged.

Archive byte checks use the original verified canonical leaf bytes, not a
serialization with defaults filled in. The message codec owns canonical singleton-array
validation and composition, framing verified fragments without reserializing SDK
objects. Its typed read uses the same parsed leaves. Accepted input contexts retain their
original history bytes and digest. A newly prepared context made from decoded
messages uses current writer output and may therefore need a new archive root;
a reuse hint always describes the original root bytes. Pending-part comparison
retains an immutable exact-content proof after original stored-byte validation,
without changing source payloads or accepting changed part IDs, content, metadata
or explicit part timestamps.

Ordinary user-content items use that same default mechanism, including metadata,
identifiers, vendor metadata, optional URL media/download settings and cache-point
TTL. SDK public constructor aliases are consulted only where there is no matching
dataclass field declaration. Required content bytes/media type, URL, uploaded-file
identity/provider and wire discriminators are not guessed. Explicit values retain
their original validation, and ordinary content keeps its existing tolerance for
extra fields.

An omitted identifier uses the SDK's deterministic content/URL/file default; it
cannot recover a discarded custom identifier. Missing URL media type retains the
SDK's deferred inference behavior, which can still fail for an opaque URL. A
referenced-content descriptor keeps its separate ref/source/revision/provenance
validation. Removing retired-reference metadata yields ordinary text, not a
fabricated reference; an explicit malformed descriptor is still rejected.

For an existing `SessionRecord` v1 without `repository_instruction_scopes`, this
same generic rule uses the field's existing empty-tuple default. The earlier
writer fixture captured from commit
`c947aa1393b335ba2e19a973634c2938f06da3e5` is retained in
`tests/ai/fixtures/persistence/session_record_without_scope_memory_v1.json`.
An older writer may ignore new fields and omit them on a later rewrite; defaults
do not reconstruct discarded scope memory. Drain and upgrade writers sharing a
store/namespace together before using new persistent semantics. Mixed-version
writers and a write-enabled downgrade are not supported; preserve the store
rather than deleting sessions or hand-modifying their payloads.

### Observe session inactivity without consuming a result

```python
view = await runtime.sessions.wait_inactive(
    session_id, principal=principal, timeout_seconds=60,
)
assert view.active_execution_id is None
```

`wait_inactive` observes that the session had no active execution or live pending
admission at one authoritative observation point. It does not reserve inactivity:
another request can be admitted immediately afterwards. It never consumes a
result, performs terminal handoff, reconciles admission, renews a lease or
cancels an execution. `RECOVERY_REQUIRED`, `PENDING_START` and
`WAITING_DEFERRED` remain busy, not successful inactivity observations.

Every observation rechecks session permission; observing a canonical execution
also checks execution-read permission. The session and execution identities remain
bound to the requested session. Native storage revision receipts on both sides
of the pending-admission read detect concurrent admission/release, including ABA
changes that do not alter the session's public semantic revision. Local pending
membership is also checked without settling an uncertain commit. Unknown local
outcomes and backend failures remain errors.

The operation requires native session/execution repositories and the Runtime's
local observation wiring. A configured shared turn backend must additionally
provide the optional `TurnSessionReader` capability: an atomic, token-free lookup
of the existing live session ownership index. A missing display-cache pointer is
not proof of absence. Backends lacking this new capability keep their existing
operations unchanged, but this new wait fails explicitly rather than guessing.
The built-in memory backend supports it only within its existing process-local
scope; it does not become cross-process coordination.

A live owner-local worker completion wakes the observer immediately; it does not
itself prove inactivity. Remote or no-worker cases use a bounded 30-second
fallback, and completed workers in recovery/deferred states cannot cause a hot
loop. No SQL transaction or connection is held while waiting. This does not add
a cross-process notification transport or promise zero polling.

`timeout_seconds=None` waits without an overall deadline. A finite non-negative
deadline expires with `WAIT_TIMEOUT`; cancellation or timeout stops only the
observer. Borrowed persistence and clients remain open. Do not wrap a long wait
in a short I/O deadline intended for a single repository call, and keep any
host-owned lease renewal independent of the wait. Calling from inside a state
transaction is rejected because it would reuse an old snapshot across waiting.

### Look up a known session

```python
session = await history.lookup_session(session_id, principal=principal)
```

`lookup_session` uses the same tenant and owned-session discovery scope as
`list_sessions`, without enumerating pages. It first checks the existing list
permission, then checks the actual resource's read permission if the ID belongs
to the caller's scope. `None` means absent or outside that scope; it does **not**
establish that the ID is globally unused. A policy denial, dependency failure or
integrity error remains an error, never an empty result. The method does not
guess an owner or retry as another principal.

The returned `SessionView` retains the existing bounded metadata-cache and
pending-observation semantics. Built-in record caches can lag remote changes by
up to three seconds; authorization still runs on every call. This is not a
reservation, an instantaneous revocation check, or proof that no active or
pending execution exists. Keep admission and inactivity decisions on their
authoritative control paths. Existing `inspect_session` still denies missing
resources rather than returning an optional value.

### Read an execution within a session

```python
detail = await history.execution_detail(
    execution_id, session_id=session_id, principal=principal,
)
status = detail.execution.status
if detail.result is not None:
    output = detail.result.output
```

`execution_detail` checks both session and execution read authorization on every
call and rejects an execution bound to a different session. It does not require
the execution to remain the session's active turn. Missing or inaccessible
resources and mismatched bindings all raise `AUTHORIZATION_DENIED`; callers
must not retry as the resource owner.

Ordinary metadata retains the existing bounded read-cache semantics. Built-in
record caches expire observations within three seconds; a remote metadata
change can remain invisible within that window. This is not an instantaneous
session-ownership or policy-revocation guarantee. Authorization callbacks still
run on each call. An observed terminal execution is reread from its authoritative
source, and its owner is checked again before returning terminal data. If that
read no longer finds the execution, the cached terminal value is not returned.

With built-in repositories, `ExecutionDetail` uses that one execution record for
terminal metadata and its result. Custom repositories retain their authoritative
`get_result` operation; the method does not impose a cross-read snapshot on
that owner. `result` is present for `SUCCEEDED`, `FAILED` and `CANCELLED`; it is
`None` for other states, including `RECOVERY_REQUIRED`, whose diagnostic
metadata remains in `execution`. Terminal integrity and retained-turn checks
are the same as `History.result()`. Usage aggregation is not performed:
`execution.usage` is `None`, while a terminal result includes its recorded usage.

This is a read-only projection, not a reconciliation, admission, lease renewal
or terminal-handoff operation. It does not promise a single physical query for
every repository or inline large object results. Existing `inspect_execution`
remains the metadata-only operation; `result` retains its not-ready and
recovery-required errors.

### Recover an accepted request after a lost response

When the host builds input from mutable context, rebuilding that input before
retrying can conflict with the original idempotency key. Read the accepted
admission before rebuilding it:

```python
admission = await runtime.sessions.lookup_admission(
    session_id, principal=principal, idempotency_key=request_key,
)
if admission is not None:
    # Compare the caller's stable request with admission.user_input before reuse.
    execution = await runtime.executions.get(
        admission.execution_id, principal=principal,
    )
```

`SessionAdmission` contains only the execution ID and accepted canonical user
input. The query requires session and execution read authorization. It does not
start a model, reconcile a pending commit, create an input capture, or scan
session history. `None` means no retained admission was found, not permission
to bypass ordinary admission. A competing admission after this read is still
resolved by the original key's atomic admission checks.

An unresolved reservation or unavailable live admission view raises
`EXECUTION_START_UNKNOWN`; keep the original key rather than starting a new
request. Reusing a key belonging to another session conflicts. Applications must
also reject changed caller input instead of treating every use of the same key
as a valid retry. Terminal-commit runtimes query their existing live request
index; a shared coordination backend must implement `TurnAdmissionReader` for
this query. An unsupported backend reports `admission_lookup_not_supported`
without changing its existing execution capabilities.
