# Execution configuration owner tradeoffs

The first reference-only configuration implementation is based on `31a9037b`
and `e541fa7a`, integrated with typed Step payload owners in `4fe633fe`.
This note records the storage and I/O contract change, not measured performance.
No new timing, SQL-count or complete-database-size result is claimed here.

Execution and capture binding references no longer store a derived summary.
View and metadata readers obtain identity, revision, capabilities and binding
digest from the fixed complete binding owner. Removing the summary deliberately
removes the former single-row, no-contract-read view optimization. Parent/child
views share the owner-read snapshot and batch contract reads; missing contract
content is an integrity error, not a reason to trust cached derived fields.

Small and large bindings follow the same fixed-owner path. Fixed Runtime policy
owners contain the accepted effective values of mode, planning, thinking,
tool_effects, session_context, restart_checkpoints, durable_trace and conversation.
Execution stores the policy reference and only later differences from that fixed
owner. Memory scope and tool-response Asset references remain request-specific
values; their absent values are omitted. Binding and policy configuration remain
Runtime archive owners, while execution payload bodies use the Step payload owner.

References, owner metadata and readback can cost more than inline storage for a
single small configuration. Sharing across executions removes duplicate body
bytes; it does not establish that every individual row or operation is cheaper.
Writable repository initialization discovers an existing archive without creating
one; read-only recovery still uses the cold read retry boundary. Policy-aware
control and retention reads can require policy content access. These costs must
remain visible in future matched measurements rather than retaining duplicate
summaries to preserve an obsolete SQL budget.

Regression coverage distinguishes fixed-body validation, atomic owner publication,
full recovery, missing-source rejection, scope remapping and real-writer policy
snapshot/GC closure. Formal suites run in CI, not as part of this documentation.
