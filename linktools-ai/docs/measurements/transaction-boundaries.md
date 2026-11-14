# Short transaction boundaries

## Status and objective

The write-transaction work remains a measured baseline and correctness inventory.
The implemented narrow slice below only removes redundant reads when a local
terminal publication is already confirmed and retained. The source baseline is
`776caf18b1c11d1e2f2122e79748521863c0c217` (tree
`5ec1b6f1178b401edf7fbca15b05d3eeeeb33873`).

The goal is to eliminate repeated short transactions for the same business
operation, not to reach an arbitrary transaction count. Each candidate must say
who needs the data, when it must be durable, which changes must be atomic, and
whether it moves reads, adds writes, or extends lock/transaction lifetime.

## Reproduction and measured baseline

Use the repository's installed check environment and `manage.py` test helper:

```bash
TURN_IO_OUTPUT=/tmp/transaction-baseline.jsonl TURN_IO_LABEL=776caf18-baseline \
PYTHONPATH="$PWD/linktools-ai/src:$PWD/linktools/src:$PWD" python - <<'PY'
import manage
manage._run_pytest('linktools-ai',
    {'paths': ['tests/ai/test_upstream_runtime_measurement.py']},
    manage._check_environment(), 'all')
PY
```

The offline fixture runs two session turns, each with three tool calls and a final
text answer: eight model requests, six tool effects and sixteen history messages.
It uses Native SQL checkpoints/frontiers, synchronous Execution persistence,
retained conversation history, Trace off, and no turn coordination. Schema setup
and session creation are outside the measurement. The window includes worker
completion and retention release, not just the first publicly available result.
Replay must cause no additional model or tool effects; the second turn must see
the first turn's tool history.

A fresh SQLite run on 2026-11-14 used Python 3.12.14, SQLite 3.53.1,
SQLAlchemy 2.0.51, aiosqlite 0.22.1, pydantic-ai-slim 2.68.0 and
pydantic-ai-harness 0.68.0:

| Operation | Turn 1 | Turn 2 | Total |
| --- | ---: | ---: | ---: |
| SELECT | 14 | 17 | 31 |
| INSERT | 65 | 64 | 129 |
| UPDATE | 22 | 22 | 44 |
| Commit | 23 | 23 | 46 |
| Commit containing writes | 19 | 19 | 38 |
| Bound text/blob bytes | 98,276 | 101,907 | 200,183 |

Bound bytes are SQL argument traffic, not unique stored bytes or wire protocol
size. This is a small upstream synthetic SQLite case, not a server latency
benchmark or an application acceptance run. Do not compare a future candidate
with a different workload, dependency environment, Trace policy or source pin.

## Business ownership of the write boundaries

The async-owner labels in the probe account for all 38 committed write transactions:

| Business boundary | Two-turn count | Why the boundary currently exists |
| --- | ---: | --- |
| Execution admission | 2 | Durable idempotency and session admission before returning the handle |
| Attempt activation | 2 | Freeze accepted input context and claim producer authority before dispatch |
| Native model request/response | 16 | Recovery frontier before provider work and checkpoint before dependent work |
| Native run completion | 2 | SDK run outcome; not yet Runtime output/cleanup success |
| Tool admission/settlement | 12 | Before/after external effect; settlement already includes the Native event |
| Execution/session terminal publication | 2 | Result, execution status and conversation publication commit together |
| Retention closure | 2 | Only after owned work drains; current dependency holds still need checking |

The eight other commits are read-only captures/queries: per turn, session-start
admission hints, committed conversation context, the first authorized wait read,
and terminal preparation each use one transaction. All 46 transactions have
async-owner labels; the JSON still reports unattributed counters for future paths. There were no open
transactions at either measured window end. Counts are observations, not long-lived
API assertions. The fixture now records each transaction and statement's async
owner path, SQL verb/table, bound bytes and commit/rollback outcome. It emits no
parameter values or private request data. SQLAlchemy commit callbacks describe
observed commit attempts; this successful fixture is not an unknown-commit proof.

### Measured operation attribution

| Boundary label | Commits | SELECT | INSERT | UPDATE |
| --- | ---: | ---: | ---: | ---: |
| `execution.read_session_start` | 2 | 2 | 0 | 0 |
| `execution.reserve_start_checkpoint` | 2 | 6 | 11 | 2 |
| `conversation.load_session_context` | 2 | 2 | 0 | 0 |
| `execution.load_result_authorized` | 2 | 2 | 0 | 0 |
| `execution.commit_prepared_agent_attempt_checkpoint` | 2 | 5 | 0 | 2 |
| `native.commit[run_started,model_request_started;snapshot=complete]` | 2 | 0 | 10 | 2 |
| `native.commit[model_request_completed;snapshot=interrupted]` | 6 | 0 | 24 | 6 |
| `execution.commit_tool_admission_boundary` | 6 | 0 | 18 | 6 |
| `execution.commit_tool_terminal_boundary` | 6 | 0 | 12 | 6 |
| `native.commit[model_request_started;snapshot=complete]` | 6 | 0 | 24 | 6 |
| `native.commit[model_request_completed;snapshot=complete]` | 2 | 0 | 8 | 2 |
| `native.commit[run_completed;snapshot=complete]` | 2 | 0 | 4 | 2 |
| `execution.commit_terminal_checkpoint` | 4 | 11 | 18 | 8 |
| `retention.release_execution_handoff` | 2 | 3 | 0 | 2 |

The terminal label includes both its preparation read and its write transaction.
Native labels distinguish complete/interrupted snapshot state, not semantic
equivalence of their complete contents. The full JSON has per-statement table names.

### Existing effect-free tool policy (not an optimization comparison)

The default fixture remains `effect_policy="replay_safe"`. Its twelve tool
transactions are not a charge imposed on every read-only tool. A diagnostic run
of the same source with `TURN_IO_TOOL_POLICY=none` exercises the existing
explicit effect-free contract: 34 total commits / 26 write commits, SELECT 31,
INSERT 99, UPDATE 32, and 151,505 bound text/blob bytes. The eight-model/six-tool/
sixteen-history and replay assertions still pass. This changes the policy, not
the implementation, and must not be reported as a 38-to-26 optimization.

Built-in `list_files`, `read_file`, `search_files`, `read_memory` and
`search_memory` already declare `effect_policy="none"`. New effect-free calls
skip ToolOperation admission/settlement; an already recorded call still follows
its original replay decision. MCP read-only hints do not grant this contract.
Effectful settlement already atomically writes the business result and Native
tool event. A failed event publication can require a separate outcome-preserving
transaction after rollback; that recovery path must not be deleted as a duplicate.

## Confirmed terminal wait read reuse

A wait that starts after this process has confirmed a complete terminal publication
can use the same retained publication helper already used after wait/poll wakeups.
Previously the first read always reloaded the source result, even while that exact
publication was retained. The change reuses the existing helper at first entry;
it adds no cache, lifetime extension, write batching or new durable owner.

The helper still applies the current authorization policy. The wait still waits
for an owned worker to settle, checks worker failure and uses the original result
consumption/handoff path. No receipt means the original source read, including
cold/remote and unconfirmed/unknown cases. External result bodies still use their
original readers. The consumer scope is entered before choosing a receipt, so
local lifetime cleanup cannot overtake consumption.

A focused SQLite probe applies the same completed-worker, explicit dependency-hold
fixture to `fc92b8d6` and the candidate. It uses the built-in tenant policy and an
inline output. The hold keeps the real publication available; it is test setup,
not a new production hold. Counts cover `wait()`, not setup or hold acquisition:

| Case | Before SELECT / read commits | After SELECT / read commits | Outcome |
| --- | ---: | ---: | --- |
| Confirmed retained publication, success | 3 / 2 | 0 / 0 | Same output |
| No retained publication, success | 3 / 2 | 3 / 2 | Original source fallback |
| Retained publication, authorization revoked | 1 / 1 | 0 / 0 | Denied |
| Retained publication, worker failure | 1 / 1 | 0 / 0 | Original failure |
| No publication, revoked authorization or worker failure | 1 / 1 | 1 / 1 | Original rejection |

These are read-only transactions; no write commit is removed. The original fixed
8-model/6-tool/16-history probe remains **46 total / 38 write commits**, SELECT 31,
INSERT 129, UPDATE 44 and 200,183 bound bytes. This slice does not solve the write
transaction target.

One availability change is intentional: an inline result with a confirmed retained
publication can now be consumed when the redundant source-result query is
unavailable. Authorization and worker-failure checks still apply. Without that
receipt the same source failure still propagates. This is not permission to
invent a result after an unknown commit or to suppress necessary body reads.

Reproduce the retained-wait probe using the normal repository environment:

```bash
RECEIPT_WAIT_IO_OUTPUT=/tmp/receipt-wait.jsonl \
PYTHONPATH="$PWD/linktools-ai/src:$PWD/linktools/src:$PWD" python - <<'PYTEST'
import manage
manage._run_pytest('linktools-ai', {'paths': [
    'tests/ai/test_confirmed_turn_receipts_independent.py'
    '::test_wait_started_after_publication_preserves_read_guards',
]}, manage._check_environment(), 'all')
PYTEST
```

The opt-in JSON logs counters only, not parameters. Focused verification covers
receipt presence/absence, current authorization rejection, worker failure, source
unavailability, real dependency holds and the existing retention/context contracts.
The current slice passed 65 receipt/retention/exact-context tests and the fixed
probe; the final receipt test with opt-in measurement passed all 15 cases.

## Candidate boundaries and proof obligations

### Admission and activation

These are currently separated by history/metadata capture, preload/input-context
preparation and worker scheduling. There is no permission to await external input
or invoke user code inside a write transaction. A possible pure-input fast path
would have to prepare outside the transaction and atomically validate the captured
session cursor/revision with idempotency, admission, input context and authority.
It must preserve:

- cancellation or forced session close after `start()` but before first dispatch;
- accepted metadata stability while later turns see updated metadata;
- a second Runtime claiming an admitted execution before a delayed worker, with
  one provider invocation and one attempt sequence;
- same-version restore, CAS rejection and unknown-commit reconciliation.

`test_initial_attempt_capture.py` covers these distinctions. No such fast path
has been implemented by this inventory. Claiming ACTIVE during `start()` closes
the existing delayed-worker takeover window. Conversely, keeping ADMITTED until
the first model checkpoint would move the authority claim past model materialization,
model context entry and capability preparation. The current contract does not
establish that all of these steps are pure or safe to run twice. Moving pure input
preparation alone does not eliminate either durable authority boundary.

### Native run completion and Execution terminal publication

`AtomicStepPersistence.after_run` has an equality/step check for snapshot reuse,
but this measured fixture still records `run_completed` with a complete snapshot
(two INSERTs and one UPDATE per turn). The response and terminal snapshot
difference was checked with a temporary diagnostic: messages and encoded bytes
are identical, while step moves from 3 to 4. Suppressing that descriptor in a
counterfactual reduces one INSERT and about 440 bound bytes per turn, but reduces
no transaction. Message bodies are already shared. Step participates in snapshot
integrity and public events, so this is not a proposed production change. Runtime subsequently reads the run
and checkpoint, checks unresolved tools, validates output and exits model and
workspace contexts. Those steps may fail or be cancelled. Native completion is
therefore not interchangeable with Execution success.

A staged built-in-only terminal plan would need an explicit, exact publication
contract for both successful and failed Runtime outcomes. It cannot report a
staged record as durable, hide it from a supported observer, or guess that output
validation/cleanup succeeds. Unknown commit, deferred-tool and error paths must
retain their own correct outcomes. No staged plan is implemented here. The existing Native commit contract confirms
snapshot state and then emits `SnapshotSavedEvent`; replacing that commit with
staging would announce uncommitted data as durable. Shared lifecycle registration
only prepares event identity/clock, and its eligibility check does not exclude
`on_event` consumers. The generic capability terminal buffer verifies its own
operation receipt, not Native `StepWriteReceipt`, and supplies no pending Native
read view. A new staged view/confirmation/exit protocol is therefore not included
as a nominally small transaction reduction.

### Terminal preparation read

The successful Native-reference fixture spends one read-only transaction and one
SELECT per turn preparing conversation publication, then opens the terminal write
transaction. The observed head determines chunk positions (first turn `[0, 8]`,
second turn `[8, 8]`), target counts, quality and replay validation. Preparing those
values from assumed absence would be incorrect. Moving all preparation inside the
write transaction would move work and transaction lifetime rather than prove a
net improvement. Separating pure body preparation from final metadata resolution
would affect the transcript, checkpoint and terminal owners plus UNKNOWN handling;
no such refactor is included merely to remove these two read-only commits.

### Necessary boundaries and non-goals

- Do not span a model request, tool execution, user hook or external I/O with a
  database transaction.
- Do not remove a model-response checkpoint before dependent tools or validation.
- Do not merge tool admission with settlement across the external effect.
- Do not remove the producer/CAS fence because a guard UPDATE leaves the logical
  execution record unchanged. Native publication's guard already runs in the
  same group transaction as its event/snapshot writes; it is not an extra
  standalone transaction. Replacing it with a prior read loses serialization
  against concurrent cancellation or producer replacement.
- Do not close retention before late dependency holds and producer drain settle.
- Keep `auto_recover`, split-store behavior, authorization, cancellation and
  observation visibility unchanged. Useful small query projections remain
  allowed; this work does not require a new model-fact storage owner.
- Keep broader IO and designated application validation separate from this
  upstream probe. Do not claim they pass because this fixture passes.

A candidate is deliverable only after matched measurement, focused correctness
checks, independent review and natural CI for its exact commit. A lower count
without those semantics is not an optimization.

This inventory does not prove 38 is a universal lower bound. It finds no safe
write-transaction reduction within the examined unchanged contracts. The Draft remains
open for a concrete candidate with measured benefit; broader workload attribution
and server/application validation remain independent next steps.
