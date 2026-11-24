# Whole-turn database I/O

**Status:** Proposed (P0). This document describes the target, not current behavior.
The target is a retained turn in an existing session with Trace disabled. ai-web
is the reference consumer of the SDK contract, not a separate persistence path.

## Contract and ownership

| Boundary | Durable responsibility | Does not belong here |
| --- | --- | --- |
| Admission | Authorize the session, identify the request, claim writer authority, retain accepted input and stable execution identity before acknowledging the request. | Reading the same session again at each API boundary; a separate read just to check whether a new request key exists. |
| Execution | Keep the admitted context and the evolving transcript in the current owner's memory. Commit only where the configured recovery or effect contract requires a durable fact. | Per-token writes, repeated ownership reads, and persistence driven by a UI subscriber. |
| Publication | Atomically add the final turn and needed history linkage, set terminal state, and release a state-owned slot when the ownership contract allows it. | A second read to confirm a known successful commit; a viewer writing terminal state. |

Session owns membership, access and chronological linkage. Execution owns the
accepted request, its durable completion and the elected writer generation.
Step owns retained messages and effect receipts, and Trace owns optional diagnostics.
A fact has one durable owner; use its reference rather than copy its payload
into another owner's record. Compose existing owner operations in a storage
transaction; do not add a separate turn store or a Runtime-root API just to
bundle calls.

### Admit

1. Read one coherent, authorized context: session/ACL, history needed for the
   next request, and any required resume pointer. Use bounded history ranges and
   batch the read; “one context” does not require one giant SQL join. Carry that
   context into the worker instead of reloading it. Reuse a preflight capture
   only if its revision is proven at claim time, or capture at admission.
2. Atomically record the accepted input, execution identity, request key and
   writer generation in the first write transaction. An existing key is an
   idempotent replay only within the same tenant/session and for the original
   request contract; otherwise return a conflict. A different concurrent
   request follows the existing busy semantics. Do not pre-read merely to
   implement uniqueness; read just the necessary receipt on conflict.
3. Acknowledge only after commit. An accepted but undispatched request must be
   discoverable from its durable admission if the process stops between commit
   and dispatch. Reuse the existing handoff/recovery entry point; do not assume
   an in-memory wake is durable, enable default recovery scanning, or add a
   synchronous outbox write solely for UI notifications.

### Run and recover

- Stay in memory for ordinary model output, intermediate text/reasoning and
  explicitly replay-safe tools. A restart may regenerate work from accepted
  input when that execution's recovery contract permits regeneration. This is
  not a promise to reproduce provider output or avoid another model charge.
  A tool with unknown replay semantics is not assumed replay-safe.
- Save a checkpoint when required to preserve continuation, approval/suspension,
  ownership handoff or external effects. Publish the effect's durable intent
  and stable invocation key before calling it. Record the outcome at the
  existing effect owner; group writes only where the outcome is already known
  and the durability boundary is unchanged. A lost response is not proof of
  failure. Reuse the same key if the external service supports it; otherwise
  reconcile uncertain outcomes and do not automatically repeat the action.
- Reuse current coordination rather than adding a second lock. A lease gives
  local validity and cancellation signals, not proof of write authority. On a
  durable boundary, enforce the applicable writer generation and expected
  state in the write predicate in the committing store; a rejected write stops
  the old owner. Never claim that a Redis/file lease check alone fences SQL.
  Keep external authority or existing recovery modes on their current safe
  path unless a takeover makes the previous generation ineligible at the
  durable boundary and that race is tested. No pessimistic database lock or
  transaction spanning model/tool I/O. Cancellation follows the existing
  owner/state transition and must not be lost at publication.
- Today the native final model-response checkpoint precedes output validation
  and terminal publication. Do not simply remove that checkpoint or defer it
  to a later hook. Retain the boundary for exact continuation. To reach the
  fast-path target, use only an explicitly selected regeneration contract and
  verify the accepted input, effect safety, output validation and observable
  callbacks. Leave the default precise-continuation contract intact; count its
  checkpoint separately. Do not add a public toggle before confirming its
  necessity and contract. See
  [Native turn boundaries](atomic-native-step-boundaries.md).

### Publish and observe

- Finish with one conditional business commit for the validated reply,
  ordered retained content, history linkage and terminal state. For SQL-owned
  authority, settle the slot in that transaction only after owned work and
  handoffs are settled. A distinct adapter lease-release call is coordination,
  not truth about the durable result. Never let an observer determine retention
  or a task's durable handoff by closing a stream.
- Publish terminal success only after a known successful commit, with the
  durable turn identity and enough information to reconcile the live timeline.
  Otherwise use the existing targeted readback for an unknown outcome, not an
  unconditional history fetch. If the final projection differs from live
  deltas, send the correction from the confirmed result. The connected page
  continues to display turn state in place without a mandatory final GET.
- Each subscriber still gets authorization, including for a fallback read.
  Reuse the existing event-driven observation contract. Capture a cursor
  before stitching backlog/live events, merge by existing event identity,
  and reconcile once from authoritative state on a detected gap or reattach.
  Overflow marks a gap or closes the stream; process death after commit may
  lose notification, so reconnect or explicit refresh must still find the
  committed turn. A silent lost notification is not guaranteed to wake an
  already attached client. Reconnect or refresh does not write the turn. Multiple tabs may require
  subscriber-specific authorization/reads; they must not multiply producer
  reads/writes or trigger repeated full-session fetches. Do not introduce a
  mandatory message broker, per-token durable stream, or per-tab DB polling.

## Implementation order

1. Land a reproducible ai-web whole-turn measurement in the existing test
   suite; keep the SDK-only [benchmark](runtime-io-benchmark.md) for diagnostics.
   Measure each boundary, not only aggregate SELECTs.
2. Consolidate the authorized admission capture and terminal projection.
   Remove redundant worker/observer reads only after testing the context and
   permissions they carry. Move lifetime/retention completion out of the
   observer path where it is still observer-triggered. Cover both live
   publication and reconnect fallback.
3. Make admission and final publication the two business commits of the
   regenerate-safe path. Reuse and extend the existing prepared-write
   composition instead of creating competing lifecycle APIs. Coalesce
   independent writes only at the same durability boundary. Keep the required
   strict-continuation/effect barriers and characterize them separately.
4. Repeat matched measurements, exercise concurrent independent sessions and
   update the verification budget. Retain the old measurement as the control.
   No deployment, downstream rollout, or assumed schema migration is part of
   this design commit. If implementation changes the public or durable
   contract, update its readers, writers, schema and fixtures together and
   document the concrete upgrade requirements.

## Acceptance and evidence

An upstream synthetic sample at `927cea3` used SQLite, a deterministic local
model, a warm existing session, and actual ai-web ASGI routes. The window
included POST, streaming, terminal read and owned cleanup, not browser JS or
cold start. Three repeats per case, one tab:

| Scenario | SELECT | All cursor executions (includes BEGIN) |
| --- | ---: | ---: |
| Trace off, plain turn | 28 | 80 |
| Trace off, one tool | 29 | 81–83 |
| Trace on, plain turn | 52 | 147–148 |
| Trace on, one tool | 78 | 223 |

The plain Trace-off variant recorded 20 INSERT, 13 UPDATE and 19 BEGIN in
each repeat. These are diagnostic baseline counts, not a SQL lower bound,
production MySQL numbers or expected gains.

- **Normal path budget:** one batch of required context reads; two business
  write commits (admit and publish) for the regenerate-safe no-effect case;
  no DB work caused by output fragment count; no mandatory terminal readback
  after known commit. Adding a viewer with identical input and transport
  must add no turn DML or producer transactions. Count lease health and
  cleanup operations separately, including them in full-path totals.
- **Correctness:** duplicate same/different input, concurrent turn on the
  same session, isolated tenants, unauthorized observer, cancellation racing
  with final commit, and a stale writer after takeover. Inject a crash after
  durable acceptance but before dispatch, between model response and
  checkpoint, on output validation failure, on effect response ambiguity,
  during an unknown final commit, and after commit but before notification. Assert one
  authoritative result, no repeated unsafe effect, and correct recovery.
- **Full-path measurement:** compare at identical SDK/DB/profile for plain
  and single-tool turns, Trace off/on, 0/1/2 observers, cold vs warm, and
  reconnect. Label intentional history reads separately from “no observer”;
  browser measurements count the actual requests and SSE connections.
  Record read shapes, DML/batch calls, begin/commit/rollback, database
  round trips where observable, bytes and p50/p95 under stated concurrency.
  Keep startup/schema probes separate. Repeat on the supported SQL
  backend(s) used to make performance claims; do not extrapolate SQLite
  numbers to remote MySQL or conflate cursor executions with round trips.
