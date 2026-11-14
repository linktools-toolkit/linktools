# Short transaction boundaries

## Status and objective

This is a measured baseline and correctness inventory, not a transaction-reduction
implementation. The source baseline is
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

The eight other commits are read-only captures/queries. Four are labeled session-start
or terminal preparation; four still have no owner label. The JSON exposes these
unattributed transactions rather than guessing their owners. There were no open
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
| `unattributed (read only)` | 4 | 4 | 0 | 0 |
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
has been implemented by this inventory.

### Native run completion and Execution terminal publication

`AtomicStepPersistence.after_run` has an equality/step check for snapshot reuse,
but this measured fixture still records `run_completed` with a complete snapshot
(two INSERTs and one UPDATE per turn). The response and terminal snapshot
differences require separate investigation; they must not be assumed redundant. Runtime subsequently reads the run
and checkpoint, checks unresolved tools, validates output and exits model and
workspace contexts. Those steps may fail or be cancelled. Native completion is
therefore not interchangeable with Execution success.

A staged built-in-only terminal plan would need an explicit, exact publication
contract for both successful and failed Runtime outcomes. It cannot report a
staged record as durable, hide it from a supported observer, or guess that output
validation/cleanup succeeds. Unknown commit, deferred-tool and error paths must
retain their own correct outcomes. No staged plan is implemented here.

### Necessary boundaries and non-goals

- Do not span a model request, tool execution, user hook or external I/O with a
  database transaction.
- Do not remove a model-response checkpoint before dependent tools or validation.
- Do not merge tool admission with settlement across the external effect.
- Do not remove the producer/CAS fence because a guard UPDATE leaves the logical
  execution record unchanged.
- Do not close retention before late dependency holds and producer drain settle.
- Keep `auto_recover`, split-store behavior, authorization, cancellation and
  observation visibility unchanged. Useful small query projections remain
  allowed; this work does not require a new model-fact storage owner.
- Keep broader IO and designated application validation separate from this
  upstream probe. Do not claim they pass because this fixture passes.

A candidate is deliverable only after matched measurement, focused correctness
checks, independent review and natural CI for its exact commit. A lower count
without those semantics is not an optimization.
