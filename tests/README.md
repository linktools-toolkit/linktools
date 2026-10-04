# Test coverage

Use `manage.py check` for the architecture, lint, compatibility, and test gates:

```bash
python manage.py check                           # merge acceptance (default)
python manage.py check linktools-ai --test-tier daily
python manage.py check linktools-ai --test-tier merge
python manage.py check linktools-ai --test-tier all # explicitly include manual probes
```

`daily` includes all unmarked tests, including new tests/packages. `merge` adds
backend and recovery combinations. `all` adds manual scale and repeated-stress
probes; it is not the release default. Direct pytest uses the same `--test-tier`
option and defaults to `merge`, but does not replace the other `manage.py` gates.
Tier exclusions happen during collection and are not test skips. Use
`PYTEST_ADDOPTS=--collect-only python manage.py check linktools-ai --test-tier all`
(or `daily`/`merge`) to inspect coverage; xdist execution summaries may omit
deselection counts. Selecting `all` does not make missing backend or optional
dependency prerequisites available.

Only two markers change scheduling:

- `merge`: additional combinations with representative automatic daily coverage
- `manual`: scale/repetition probes that require explicit `--test-tier all`

Unmarked tests or unrelated markers stay in `daily`. A `manual` mark takes
precedence if a case has both marks. Do not classify tests only by duration.
Unique cancellation, corruption, recovery, external-effect and regression
obligations must retain automatic coverage.

## CI selection

| Event | Coverage |
| --- | --- |
| Draft PR opened, synchronized, reopened or converted to draft | `daily` |
| Ready PR opened, synchronized, reopened or made ready for review | `merge` |
| Push to `master` | `merge` |
| Release checks through `workflow_call` | `merge` |
| Manual `Python checks` dispatch | Explicit `daily`, `merge` or `all`; default `merge` |

The reusable workflow intentionally declares no tier input. It cannot accept an
`all` choice from a caller; the publishing workflow's manual release-version
input does not select manual tests. No path, ready transition, release or timer
automatically enables manual probes. There is no scheduled run.

Both Python versions and automatic package discovery are retained. Existing
matrix check names are stable. The `Python test coverage` aggregate fails unless
discovery, compatibility, and the whole package matrix actually succeed; skipped
jobs do not count as completed coverage. Its summary states the selected tier.
Repository protection settings are managed separately.

## Coverage responsibilities

| Family | Daily | Merge additions | Explicit manual additions |
| --- | --- | --- | --- |
| SQLite task concurrency | One serial DAG and one concurrent fan-in DAG; existing CAS/readback/cancel/failure/reopen regressions | None | Original 20 serial + 20 concurrent iterations |
| Captured graph inputs | Eight pairwise-balanced combinations across declaration/materialized, clean/captured, changed/deleted and fixed/reproject | Remaining eight combinations, completing all 16 | None |
| Process-crash recovery | All five local SQLite boundaries; tool-completed boundary on SQLAlchemy SQLite and split storage; dedicated failure regressions | Checkpoint, projection and terminal boundaries on SQLAlchemy SQLite and split storage, completing 3 backends × 5 boundaries | None |
| Server percentile compatibility | Empty/singleton inputs and existing small native-dialect aggregate/selection contracts | All original 100-sample nearest-rank floating-point boundaries, grouped and ungrouped | None |
| Server aggregate query plan | 100-row real PostgreSQL/MySQL result and EXPLAIN/source-read checks | None | Original 100,000-row plan/source-read checks |
| SQLite measurement round trips | 100 observations, every supported aggregation and bounded-statement assertion | None | Original 1,500 observations across batching thresholds |
| SQLite extrema plans | 100 observations, MIN/MAX values, one source scan/JSON expansion, no order sort | None | Original 1,500-row EXPLAIN checks |
| SQLite record validation | 100 observations, both metric sources with/without filters, each row validated once and one SQL statement | None | Original 1,500-row validation/statement checks |
| SQLite integer totals | 100 × INT64_MAX, exact overflowing integer sum/type and one statement | None | Original 1,500-row exact aggregate |
| SQLite floating reduction | 100 observations, chronological reduction and one recursive aggregate statement | None | Original 1,500-row reduction |

Pagination, scan/group limits, numeric-order edge cases, and corruption semantics
are not moved merely because they contain large counts. Native database fixtures
remain isolated; daemon scope is unchanged.

## Shared preparation

Consolidations reduce preparation, not assertions:

- Scoring-selection comparisons share one evaluation/rescore, then independently
  assert rescore/rule/failing selection, saved cutoff and persisted report. Each
  live-usage monkeypatch is scoped to its assertion group. Unknown-target usage
  and unselected-candidate regressions remain separate.
- Server percentiles seed each of 0/1/100 observations once per dialect, then
  exercise every original percentile in both query layouts.
- Each corrupted server record exercises all three metric query paths. Distinct
  corruptions have separate fixtures; snapshots/writers/pruning remain isolated.
- Inline capture adapters share an accepted source only when projected/file-change
  inputs match. Both captures are made before mutation and use separate durable
  dataset/idempotency identities. All original adapters, fixed/reproject modes,
  binary/file ordering and attachment assertions remain.
