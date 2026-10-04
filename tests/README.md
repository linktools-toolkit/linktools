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

Both Python versions and automatic package discovery are retained. Non-AI
packages keep one check per version. AI runs two disjoint file-family groups per
version: `evaluation` selects `test_evaluation*` and filenames containing
`capture`; `runtime` selects every other file under `tests/ai`. New files join a
group automatically; no file manifest or recorded timing database is required.
Grouping never changes tier eligibility. The default group is `all`, so local
checks continue to cover the complete chosen tier. To run one group locally:

```bash
PYTEST_ADDOPTS='--ai-group=evaluation' python manage.py check linktools-ai --test-tier daily
PYTEST_ADDOPTS='--ai-group=runtime' python manage.py check linktools-ai --test-tier daily
```

The original `Python <version> linktools-ai checks` names remain as aggregate
checks. They conservatively require the entire package matrix to succeed, so a
failure in another package/version also fails both AI aggregates. The `Python
test coverage` aggregate requires discovery, compatibility, all package groups
and these aggregates to succeed; skipped/failed jobs or empty test groups do not
count as completed coverage. Its summary states the selected tier. Repository
protection settings are managed separately.

With the current five packages this uses 17 jobs instead of 13: two additional
AI execution jobs and two lightweight aggregates. Each AI group installs the
same dependencies and runs the package architecture/lint gates before its own
tests; files stay intact for fixture reuse and four-worker `loadfile` scheduling.
The measured evaluation/capture family accounts for about 56% of cumulative
daily test-call time, motivating two groups. That is not a wall-time prediction;
extra runner startup, installation and gate work increase total runner usage.

## Coverage responsibilities

| Family | Daily | Merge additions | Explicit manual additions |
| --- | --- | --- | --- |
| SQLite task concurrency | One serial DAG and one concurrent fan-in DAG; existing CAS/readback/cancel/failure/reopen regressions | None | Original 20 serial + 20 concurrent iterations |
| Inline workspace captures | Four adapter scenarios, each running fixed input then reprojection against one accepted source (8 evaluation runs) | Four complementary adapter/projected/file-change scenarios, completing the original 16 evaluation runs | None |
| Scorer request usage | Initial scoring with known usage; rescoring with unknown usage; complete inexpensive report-gate truth table | Initial scoring with unknown usage and rescoring with known usage; all four retain actual usage, saved cutoff and report assertions | None |
| Recovery bootstrap | Activated-before-first-response and unconfirmed-effect process exits; existing cancellation, live-claim and split-handoff regressions | Request-checkpoint exit and a second crash during recovered terminal completion | None |
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
- Each inline capture scenario keeps fixed-input replay followed by reprojection
  on the same accepted source. The four daily scenarios cover all four adapters,
  direct/projected inputs and changed/deleted files, with eight evaluation runs.
  Merge adds the complementary four scenarios, preserving all sixteen original
  evaluation runs and attachment/prompt assertions. Explicit scenario selection
  requires eight source preparations in merge instead of the earlier four shared
  preparations; daily still prepares four sources. This trades some merge setup
  reuse for a smaller, visible daily integration set without hidden mode skips.

The initial/rescore × known/unknown scorer-usage integration matrix stays complete
in merge. Daily keeps the normal initial-scoring path and the rescore unknown-usage
failure path; the report-only completeness/requirement truth table always runs.
Known-usage rows use one equivalent successful case per candidate. Unknown-usage
rows retain both successful and failing cases, so 50% valid score coverage still
passes the permissive gate while unknown usage independently blocks the strict
gate. Saved reports and persisted usage cutoff assertions remain in every row.
Recovery request-checkpoint and repeated-crash interactions likewise stay in merge,
while unknown effects, cancellation, live claims, split ownership and real process
exit safety retain daily coverage. Test counts alone do not measure these changes:
one collected inline scenario performs two complete evaluation runs.
