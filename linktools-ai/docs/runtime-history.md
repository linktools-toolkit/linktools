# Reading execution content while a run is active

Observation events identify facts; the corresponding Runtime domain owns their
content. Receiving a boundary event does not require waiting for another model
request before reading the confirmed content in the **same Runtime instance**.

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

An empty page means no matching fact is currently available. A result item with
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

Only confirmed parts are included. Streaming text deltas are live updates, not
complete history parts. `transcript` contains user and assistant text; thinking,
tool arguments and tool returns belong to `history`.

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
`None` while the live event has no durable coordinate, including live deltas.
None of these coordinates are interchangeable.

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

The Runtime reads the existing process-local transcript staging and then the
execution archive. Confirmed parts are staged before their notification is
published. Completed messages retain their normal transcript coordinates. An
in-progress tail message freezes its visible part identities when a page is
started: parallel tool completion order can differ from final message order.
Continuing that cursor after more results arrive or staging is released reads
the original set exactly once. Start a fresh query to see newer facts.

This does not promise immediate readback from another process or after a
restart. Those readers see what the configured execution archive has already
materialized. No new durable content journal, database migration, or
cross-process event/content transaction is introduced.

Bodies are opt-in in content-bearing read responses; trace remains metadata-only. Model interaction and attachment-metadata reads
do not resolve unrequested bodies. Existing archived history/transcript
projection may still decode message chunks to identify items even when
`include_content=False`; it does not return those bodies to the caller.


Model interaction cursors freeze request identities and high-water marks, not
request lifecycle state. A captured RUNNING request can be terminal when a later
page reads it. Refresh without a cursor to discover newly started requests.
Model interaction reads can see confirmed process-local staging; aggregate
`usage()` reads archived usage and may lag the active request view. Check its
completeness metadata rather than treating an incomplete total as final.

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

`ModelInteractionReadBoundary.cutoffs` bounds visible identities per Agent run.
`durable_cutoffs` separately bounds archived identities; a terminal staged row
is not necessarily archived yet. `local_staging_available` is conservative:
the current run must actually exist in this recorder. It is false before local
run admission, for a remote producer without local run ownership, and after
local staging is released. `durable_history_available` reports the configured
retained archive's availability, not a claim that in-memory storage survives
restart. Disabled or unavailable history is explicitly marked; authorization
and integrity failures are not converted into empty successful reads.

Metadata reads return a bounded tuple of contiguous request identities, always
with `content_included=False`, an empty request and no response. They do not
resolve content payloads. Depth is relative to the selected execution (zero);
the graph observer supplies its known tree-relative depth. To reread one active
request at sequence N, use `after_model_request_seq=N - 1` and
`through_model_request_seq=N`. Reusing an old history page cursor cannot detect
a completion for an identity already passed by that cursor.

Staging is captured before archive reads; the archived terminal fact wins a
handoff overlap. The fixed request cutoff does not freeze lifecycle status.
Missing identities inside a declared range, inconsistent ownership, and
conflicting terminal metadata raise integrity errors rather than fabricating
coverage. These reads do not create a second lifecycle store or guarantee
cross-process visibility of an uncommitted start.

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
