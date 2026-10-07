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

## Read one tool call

The default watch keeps safe locator and status metadata, including
`agent_run_sequence` and `call_id`, without exposing arguments or results.

```python
payload = event.event.payload  # ExecutionTreeEvent from execution.watch()
page = await runtime.executions.history(
    event.execution_id,
    principal=principal,
    agent_run_sequence=payload["agent_run_sequence"],
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
    agent_run_sequence=payload["agent_run_sequence"],
    message_sequence=payload["message_sequence"],
    part_index=payload["part_index"],
    include_content=True,
)
```

`agent_run_sequence` limits the query to that exact execution and run. All selectors are optional and combine with AND; they may also be used
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

A model request is identified by `(execution_id, agent_run_sequence,
request_sequence)`, exactly the identity returned by `ModelInteractionItem`.
`agent_run_sequence` is a one-based run sequence within an execution, not a
session turn or child-execution ID. `request_sequence` is a one-based logical
model-request sequence within that run. Its journal includes agent requests,
output-correction retries and compaction requests; it is not a provider request
ID or a count of transport attempts. The sequence continues from recorded
requests when the same run is recovered.

`step_index` is the SDK's non-negative execution-step index within the run.
Several requests can belong to the same step. It is useful for grouping, but
cannot replace the request identity. `message_sequence` is the one-based raw
transcript message coordinate (`ExecutionHistoryItem.sequence`), and
`part_index` is the zero-based part coordinate inside that message. A trace
item's `sequence` instead locates its step event; none of these coordinates are
interchangeable.

For assistant response parts and tool calls, history's `request_sequence`
identifies the request that **produced** the response. For tool returns and
call-specific retry prompts it identifies the request that **initiated the
call**, even when a later model request carries that return. History's
`step_index` is the originating model request's step. The existing successful
model step event records the actual raw response's `message_sequence`; trace
exposes that coordinate on its model response item. Tool trace events retain
their own occurrence step and the originating request sequence.

User/system messages, unassociated retry prompts, incomplete response parts,
and facts without a recorded association have `request_sequence=None` and
`step_index=None`. No association is guessed from time, message order, or
matching text. Compaction requests appear in model interactions and trace but
do not invent a response in conversation history. To inspect all inputs carried
by a request, read its `ModelInteractionItem.request`; history is an occurrence
view, not a duplicate of every model input context.

Both `history(...)` and `trace(...)` accept `agent_run_sequence`,
`request_sequence`, `step_index`, and `tool_call_id`. History additionally
accepts `message_sequence` and `part_index`. Without a run selector, a request
or call selector matches each run in the normal query scope; always retain
returned execution/run coordinates to disambiguate results. For example:

```python
page = await execution.history(
    agent_run_sequence=1, request_sequence=2, include_content=True, limit=20,
)
trace_page = await runtime.history.trace(
    execution.execution_id, principal=principal,
    agent_run_sequence=1, request_sequence=2, limit=20,
)
# Continue with the same selectors and content mode.
page = await execution.history(
    agent_run_sequence=1, request_sequence=2, include_content=True,
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

Bodies are opt-in in responses. Model interaction and attachment-metadata reads
do not resolve unrequested bodies. Existing archived history/transcript
projection may still decode message chunks to identify items even when
`include_content=False`; it does not return those bodies to the caller.


Model interaction cursors freeze request identities and high-water marks, not
request lifecycle state. A captured RUNNING request can be terminal when a later
page reads it. Refresh without a cursor to discover newly started requests.
Model interaction reads can see confirmed process-local staging; aggregate
`usage()` reads archived usage and may lag the active request view. Check its
completeness metadata rather than treating an incomplete total as final.
