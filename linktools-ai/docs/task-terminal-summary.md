# Task terminal aggregation

The Task repository keeps derived status counts in the existing `task_graph`
record's `task_summary` envelope extension. Node payloads remain authoritative.
The summary is not a second scheduler, status index, or public Task API.

The projection counts each TaskStatus and separately counts FAILED
and BLOCKED nodes whose definitions use the `propagate` failure policy. Only
nonzero status counts are written; omitted status keys mean zero. The reader
also accepts full maps including zero counts. Graph
status uses the same precedence for both complete-state and incremental paths:
RECOVERY_REQUIRED; RUNNING or WAITING; PENDING or READY; propagated FAILED;
propagated BLOCKED; CANCELLED; otherwise SUCCEEDED. Explicit cancellation keeps
its existing graph validation and reconciliation semantics.

## Writers and transaction boundary

All current Task writers maintain the projection in the node/event transaction:

| Operation | Summary handling |
| --- | --- |
| Admission | Build from definitions and initial node states. |
| Claim, execution handoff | Apply the target's status delta. |
| Ordinary completion | Read header/summary, target state, target definition and result slot together; apply the target delta. |
| Failure, expansion | Keep complete-state validation and rebuild counts. |
| Retry, scheduler reconciliation, cancellation, recovery | Rebuild from the complete state already required by the operation. |
| Execution binding, prepared input, cancellation registration | No status change; preserve the projection. |
| Lease renewal | No status change; no summary write. |
| Admission projection repair | Guard the original graph revision, read complete state and rebuild. |

A graph replacement carrying the summary is itself the existing optimistic CAS,
using the revision captured with the original header. It does not first guard
and then update the same header again. Semantic mutations that do not change the
summary or header still guard the graph. Node/result/event writes remain in the
same transaction, so rollback, cancellation and unknown commit/readback keep
their existing durable-outcome semantics. No additional write transaction or
process-local mutable-state cache is introduced.

## Validation, restore and upgrade

Ordinary completion trusts the current owner-maintained summary and validates
its freshly read target records. It does not scan unrelated nodes to defend
against direct database changes outside the Task owner contract.

Full graph reads continue decoding and validating all source node payloads and
use those payloads for the returned state. At consistent read boundaries they
also compare the derived summary with complete node state. A nested writable
READ COMMITTED transaction can read an old header followed by newer node rows;
it must not diagnose that lawful interleaving as corrupted summary data. Full
mutations validate the previous summary only after their captured graph CAS
succeeds, then publish the rebuilt projection. Unchanged scheduler reads do not
write or add a separate freshness protocol.

Snapshot restore validates a present summary against complete imported nodes
and definitions. Snapshot record encoding retains the summary bytes, so the
existing snapshot integrity digest covers them. There is no independent summary
checksum. Cleanup follows the existing graph record lifecycle.

Existing records without the summary remain readable. The first status-changing
owner operation can rebuild it from complete source state. A malformed present
summary, including explicit null, is not treated as an absent projection.

This does not support mixed old/new writers. The existing coordinated upgrade
contract in [Runtime history](runtime-history.md) and
[Runtime capabilities](runtime-capabilities.md) requires draining and upgrading
all writers that share the same store/namespace. An old writer could preserve
counts it does not maintain; write-enabled downgrade remains unsupported.

## Cost boundary

Ordinary successful completion reads only the graph header, target definition
and target state (plus an existing result on idempotent readback). Across N
independent successful nodes, completion record reads are linear rather than
N(N+3). Claim concurrency checks, scheduler reconciliation, failures, expansion,
full public reads and restore retain their necessary broader reads. This is not
a claim that the whole scheduling workload is constant-cost.

Measurements separate SQL statements, batch parameter rows, source record rows,
raw JSON text bytes, parsed canonical JSON bytes and transaction commits. The
small summary adds bytes to header reads, and small graphs may show less or no
whole-operation benefit. Fixture measurements do not replace consumer
acceptance or real MySQL concurrency verification.
