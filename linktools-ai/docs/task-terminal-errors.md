# Confirmed Task execution failures

A failure code describes why an execution failed. It does not by itself say
whether the execution's terminal state was durably confirmed.

`TaskNodeRunError` accepts an optional `terminal_status`, restricted to the
`ExecutionStatus.FAILED` and `ExecutionStatus.CANCELLED` enum values. A Task
runner supplies it only after successfully reading an authoritative terminal
result or receiving a confirmed terminal commit result. The default `None`
preserves the existing treatment of uncertain execution/readback outcomes.

The scheduler uses confirmed terminal status to settle a node, even if the
execution failed with a code also used for temporary storage uncertainty or
concurrent settlement. It preserves the original error code and details.
A confirmed cancelled execution settles as cancelled rather than failed.

This evidence exists only in memory. Recovery reconstructs it from the original
execution result, including records created before this parameter existed.
No schema migration or durable format change is required. An unavailable
confirmation read remains unconfirmed and continues through recovery. Task
retry limits and non-replay-safe external-effect recovery are unchanged;
confirming a failed execution does not rerun its body.
