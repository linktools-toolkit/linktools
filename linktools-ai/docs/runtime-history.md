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
| Request attachment inclusion | execution ID, agent run sequence, request sequence, attachment ID | `runtime.executions.attachment_facts(...)` |
| Task result | graph ID, task ID, result reference | the task run's `result`, `result_ref`, or `results` operation |

Model interactions retain their existing recursive execution scope. Ordinary
history and trace retain their existing root/direct-child scope; transcript
retains its selected-execution/current-run scope. A recursive watch does not
expand these query scopes. Read a deeper subagent using that event's own
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
`content_included=False` means the caller did not request the body. Tool status
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

`agent_run_sequence` limits the query to that exact execution and run. Tool and
message selectors require it, and a part selector also requires a message
selector. Different runs and child executions may reuse a call ID. Selectors,
execution identity, content mode, and tenant are bound to the cursor.

Only confirmed parts are included. Streaming text deltas are live updates, not
complete history parts. `transcript` contains user and assistant text; thinking,
tool arguments and tool returns belong to `history`.

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
