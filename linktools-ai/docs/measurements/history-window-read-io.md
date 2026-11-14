# Exact History window reads

## Scope

Baseline: `52a9edb6ebee3d197edc10e7d00baf45cb19fa43`.
The isolated synthetic fixture performs two deterministic model requests and
twelve effect-free tool deliveries. It then requests the final tool's exact
History projection with `agent_run_seq=1`, `model_request_seq=1`, and `limit=2`.
Both content modes must equal the corresponding items in the complete History.

**The measurements below are for one exact History query, not for executing the
whole fixture.** They are SQLite SQL attempts and transaction events, not disk
flushes, network bytes, or MySQL latency. Initialization and the full History
reference query are outside the measurement window. A task-local context marker
includes database work owned by the measured query and excludes earlier worker
cleanup. The optional listener stores counts only, without SQL parameters.

## Minimal change

`TranscriptRepository._read_range_start` previously read a transcript head and
its seek hint separately when starting a nonempty window after message zero.
It now reads those two records together in the same existing transaction.
The head remains authoritative; the seek record remains only a hint. Missing or
malformed hints retain their existing fallback to validated transcript facts.
No public interface, durable format, cache policy, lock, or transaction boundary
changes. Empty windows and already-captured heads keep their existing paths.

| Read configuration | SELECT before → after | BEGIN before → after | Commit before → after |
| --- | --- | --- | --- |
| No ordinary cache, live warm | 12 → 11 | 7 → 7 | 7 → 7 |
| Ordinary cache, live warm | 10 → 9 | 5 → 5 | 5 → 5 |
| No ordinary cache, reopened cold | 14 → 13 | 9 → 9 | 8 → 8 |
| Ordinary cache, reopened cold | 14 → 13 | 9 → 9 | 9 → 9 |
| No ordinary cache, reopened warm | 12 → 11 | 7 → 7 | 7 → 7 |
| Ordinary cache, reopened warm | 10 → 9 | 5 → 5 | 5 → 5 |

Both `include_content=False` and `True` give the same counts. The uncached cold
reader has one rolled-back content-discovery read before opening its archive;
that existing retry is unchanged. No measured read performs a write.

A separate counterfactual grouped post-authorization execution root and child
metadata. It reduced one uncached read transaction but forced a source root
read on the ordinary-cache path, cancelling the SELECT saving there. That
counterfactual was removed rather than changing the existing cache policy.

## Verification and reproduction

The public regression covers exact item equality, order-preserving one-item
pagination, unknown tool selection, missing execution, tenant denial, both
content modes, and close/reopen with fresh caches. A positive-offset transcript
test covers present, missing, and malformed hints while requiring one bounded
head/seek metadata query. Existing range tests cover append cutoffs, sparse
windows, malformed content, paused object I/O, cancellation, and concurrent
collection. The sparse-window test no longer freezes which private helper is
called; its output, boundary, fallback and separate SQL skipped-page coverage
remain.

The final focused set passed **155 tests** through `manage.py` (40.06 seconds),
including the seven new cases. The repository's configured Ruff selections and
`git diff --check` passed. Independent static review found no must-fix in this
slice. The same new positive-offset budget check fails on the unchanged baseline
with two metadata queries instead of one; its returned messages still match.
These are upstream synthetic checks, not application or real-MySQL acceptance.

From the repository root, with the normal development environment installed:

```sh
HISTORY_IO_OUTPUT=/tmp/history-read-counts.jsonl HISTORY_IO_LABEL=current python - <<'PY'
import manage
manage._run_pytest('linktools-ai', {
    'paths': ['tests/ai/test_history_exact_read_io.py'],
}, manage._check_environment(), 'all')
PY
```

For a matched baseline, copy only the same new test file into a clean worktree
at the baseline SHA and run its public-runtime test selector. The new
positive-offset I/O-budget test is expected to fail on the baseline because
that version issues separate head and hint queries. Use the same Python and
dependency environment for both worktrees.

## Remaining boundary

This is one fewer SELECT for each affected transcript window, not a reduction
of the fixed two-turn workload's 38 write transactions. It does not claim that
exact tool selection avoids all unrelated message materialization:
`include_content=False` still removes content after History projection. Broader
query selection, whole-workload transaction costs, long-session retention,
and designated application acceptance remain separate work. Do not extrapolate
these counts to a different pin, tool policy, graph, or History query shape.
