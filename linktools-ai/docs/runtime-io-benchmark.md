# Retained-turn I/O benchmark

`tests/ai/test_official_runtime_attribution.py` is an opt-in measurement of the
normal Runtime path. Its private trace helper is shipped in `tests/ai`; no other
checkout, diagnostic plugin, or private helper directory is needed.

## Scenario and measurement boundary

One Runtime and one session execute two consecutive retained turns. Each turn
makes four local FunctionModel requests and three sequential replay-safe tool
calls, then returns `answer`. The configuration is SQLite, native
`StepPersistence` with frontier capture, synchronous commits, a 16 MiB in-memory
cache, default authorization, no `TraceCapability`, and no turn coordination.
The diagnostic wrappers below are separate from `TraceCapability`.

Each window begins immediately before `Session.start` and ends after the result,
owned worker, and retained owner cleanup have finished. Database provisioning,
Runtime/session creation, history inspection, budget-usage inspection, and
same-key replay are outside the windows. The assertions retain all eight model
calls, six effects, full model-visible history across turns, sixteen persisted
messages, six tool returns, two answers, and replay without extra model calls
or tool effects.

`TURN_IO_TOOL_BUDGET=3` optionally passes `RunBudget(tool_calls=3)` through the
public `Session.start` parameter for both turns and same-key replay. Each turn
has its own budget scope. Public `budget_usage()` checks verify four model
requests, three tools, no in-flight or unknown model usage, and unchanged complete
usage after replay of the same execution. Without this variable, the budget is
`None`, preserving the normal no-budget measurement.

## Reproduce through the repository gate

Use Python 3.11 or later, the repository's development requirements, and an
editable installation made with `python manage.py install --editable linktools-ai`.
Run the following from the repository root in that environment. The explicit
source path and import assertion prevent a different editable checkout from
supplying `linktools.ai`.

```bash
export PYTHONPATH="$PWD:$PWD/linktools/src:$PWD/linktools-ai/src"
unset PYTEST_PLUGINS TURN_IO_FRESH_BASELINE TURN_IO_ENABLE_TRACE TURN_IO_TOOL_BUDGET
export TURN_IO_FRONTIER=1
export PYTEST_ADDOPTS='-k test_official_runtime_attribution -vv'
out="$(mktemp -d)"
python - <<'PY'
from pathlib import Path
import linktools.ai
assert Path(linktools.ai.__file__).resolve() == (
    Path.cwd() / "linktools-ai/src/linktools/ai/__init__.py"
)
PY

TURN_IO_LABEL=small TURN_IO_TOOL_BYTES=4 \
  TURN_IO_OUTPUT="$out/small.jsonl" TURN_IO_TRACE="$out/small.trace.json" \
  python manage.py check linktools-ai --test-tier all

TURN_IO_LABEL=large TURN_IO_TOOL_BYTES=81920 \
  TURN_IO_OUTPUT="$out/large.jsonl" TURN_IO_TRACE="$out/large.trace.json" \
  python manage.py check linktools-ai --test-tier all

TURN_IO_LABEL=budget-small TURN_IO_TOOL_BYTES=4 TURN_IO_TOOL_BUDGET=3 \
  TURN_IO_OUTPUT="$out/budget-small.jsonl" TURN_IO_TRACE="$out/budget-small.trace.json" \
  python manage.py check linktools-ai --test-tier all

printf 'Measurement files: %s\n' "$out"
```

Each command runs the package architecture and lint gates, collects the test
suite, and selects only the named benchmark case. Check for `1 selected` and
`1 passed`; do not mistake deselected tests for passing tests. The benchmark is
skipped unless `TURN_IO_OUTPUT` is set. That JSONL file is appended to; each trace
JSON file is replaced. If `TURN_IO_TRACE` is omitted, the trace is written beside
the output with a `.trace.json` suffix. The parent directory must already exist.
The historical `TURN_IO_FRESH_BASELINE` comparison switch is not a supported
parameter for this source and must remain unset.

## Interpret the output

- `sql.SELECT`, `sql.INSERT`, `sql.UPDATE`, and `sql.DELETE` count SQL statement
  execution calls, not affected or returned database rows. SQL writes are the
  sum of INSERT, UPDATE, and DELETE calls; executemany can affect multiple rows.
- `sql.transactions` counts begins, `sql.commits` counts commits, and
  `sql.write_commits` counts committed transactions that executed at least one
  write statement. None of these is a disk-fsync or storage-device I/O count.
- Trace `rows` are statement records, linked to transaction and async call-path
  evidence. They are not database row counts. `state.source_records` counts
  decoded state records only, not all rows returned by all SELECTs.
- `sql.bind_text_blob_bytes` sums UTF-8 text and binary bound parameters for all
  measured SQL. It excludes numeric parameters, SQL text, result traffic,
  protocol framing, database pages, and disk/WAL bytes. It is not write volume.
- `state.source_record_json_bytes` measures compact JSON re-encoding of decoded
  state data. Cache and object byte counters measure the respective payloads
  passed or returned by those APIs. These overlap logically and must not be
  added into a claimed physical-I/O total.

The helper records call paths, statement/transaction attribution, native step
facts, and hashes of loaded source modules. It preserves the observed persistence
operations. Instrumentation and test collection add overhead: pytest wall time
is **not** turn latency, first-token latency, or a performance speedup result.
Exact counts are measurement evidence for this configuration, not API contracts.

## Recorded reference

The reference run used Python 3.12.14, PydanticAI 2.66.0, Harness 0.66.0,
SQLAlchemy 2.0.51, and aiosqlite 0.22.1. All three commands passed their package
gates and selected exactly one passing case. Pairs below are first/second turn.

| Quantity | 4 B, no budget | 80 KiB, no budget | 4 B, tool budget 3 |
| --- | ---: | ---: | ---: |
| SELECT calls | 13 / 11 | 13 / 11 | 29 / 27 |
| INSERT calls | 58 / 57 | 70 / 69 | 66 / 65 |
| UPDATE calls | 22 / 22 | 22 / 22 | 33 / 33 |
| DELETE calls | 0 / 0 | 0 / 0 | 0 / 0 |
| SQL write calls | 80 / 79 | 92 / 91 | 99 / 98 |
| Write commits | 19 / 19 | 25 / 25 | 27 / 27 |
| Transactions begun | 26 / 24 | 32 / 30 | 38 / 36 |
| All commits | 26 / 24 | 32 / 30 | 38 / 36 |
| Decoded state records | 14 / 17 | 14 / 17 | 33 / 36 |
| Bound SQL text/blob bytes | 107836 / 122979 | 602017 / 617159 | 129422 / 144565 |
| Cache PUT payload bytes | 32303 / 38318 | 278051 / 529814 | 45632 / 52211 |

All transaction begins in these successful runs had matching commits. Budget
accounting adds sixteen SELECTs and eight write commits per turn; the eight
commits are four model admissions plus four model settlements. A tool-only limit
still exposes model and token usage. The measured budget usage is 4 model
requests, 3 tools, and 229 / 273 settled tokens, with unchanged usage after replay.
Bound byte counts can vary with serialized metadata; retain the generated files
when comparing runs.

The source fingerprint covers all 357 Python files under `linktools-ai/src` and
was unchanged before and after the three runs:
`0ab2e6ef3e6451512edf1f5f33a03f85e47a35ae0c724278c8822536b6de9816`.
Measurement file SHA-256 values:

- `tests/ai/test_official_runtime_attribution.py`:
  `82922e052558dd237f6d3c399f4d319bbf7375a9aae2bbf12a01a3541362e382`
- `tests/ai/_official_runtime_io_trace.py`:
  `a5e1e9517b47edf8e0c3758a4f468678dd0f647330fcb34e5f9606d5d989f4e4`
- `tests/ai/_runtime_test_helpers.py`:
  `49600480c2499166c22004099117f376aece36a104bc6fbfc66ff2161c40856e`

To calculate the same content fingerprints from the repository root:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

sources = {
    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(Path("linktools-ai/src").rglob("*.py"))
}
encoded = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
print("source files:", len(sources), "source SHA-256:", hashlib.sha256(encoded).hexdigest())
for name in (
    "tests/ai/test_official_runtime_attribution.py",
    "tests/ai/_official_runtime_io_trace.py",
    "tests/ai/_runtime_test_helpers.py",
):
    print(name, hashlib.sha256(Path(name).read_bytes()).hexdigest())
PY
```

This is a deterministic workload and I/O attribution check, not the full test
suite, backend parity, process-recovery acceptance, or a latency benchmark.
