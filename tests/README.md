# Test coverage

Use `manage.py check` for the architecture, lint, compatibility, and test gates:

```bash
python manage.py check                           # merge acceptance (default)
python manage.py check linktools-ai --test-tier daily
python manage.py check linktools-ai --test-tier merge
python manage.py check linktools-ai --test-tier all # explicitly include manual probes
```

`daily` includes all collected unmarked tests, including new test files. `merge` adds
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

## Package test discovery

Each package's `checks.pytest.paths` declares any nonconventional test locations.
`manage.py check` also includes `<package>/tests` and `tests/<package short name>`
when they contain pytest's default `test_*.py` or `*_test.py` filenames. These
conventions apply to newly discovered packages without editing the CI workflow
or maintaining a package list. Overlapping paths are collected once. A package
with no declared or conventional tests may still run its architecture/lint gates;
its log explicitly reports that no pytest tests were found. Other directory or
filename conventions require explicit pytest configuration.

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

The execution plan in `scripts/check/matrix.py` owns the Python versions and
pytest execution options. Package group names and filename matching rules live
in their existing `linktools.yml` under `checks.pytest.groups`. Both execution and legacy
aggregate jobs consume that plan; automatic package discovery is retained. Non-AI
packages keep one check per version unless their manifest declares groups. For
example, `linktools-ai/linktools.yml` declares:

```yaml
checks:
  pytest:
    paths:
      - ../tests/ai
    groups:
      evaluation:
        - test_evaluation*
        - '*capture*'
      runtime: []
```

`paths` may be omitted when conventional test directories are present. A
declared pytest check with no test paths fails rather than collecting the entire
repository. Patterns are case-sensitive filename globs, not paths. Exactly one group must
have an empty list: it receives every unmatched file, including newly added
files. A file matching multiple non-default groups fails collection rather than
relying on ordering. Group names must be simple letters, digits, hyphens or
underscores; `all` is reserved for unfiltered execution. A package without groups
gets one `all` job. No Python-side package name or file-family classifier is
needed to add another grouped package.

`manage.py check` validates the manifest and passes the selected package's
resolved pytest configuration to its subprocess. Group filtering applies only
within that package's test paths, including conventional directories. The default
`all` group preserves complete local coverage; named groups must be declared in
the selected package's manifest, and unknown groups fail clearly. Tier selection
remains independent:

```bash
PYTEST_ADDOPTS='--test-group=evaluation' python manage.py check linktools-ai --test-tier daily
PYTEST_ADDOPTS='--test-group=runtime' python manage.py check linktools-ai --test-tier daily
```

The original `Python <version> linktools-ai checks` names remain as aggregate
checks. They conservatively require the entire package matrix to succeed, so a
failure in another package/version also fails both AI aggregates. The `Python
test coverage` aggregate requires discovery, compatibility, all package groups
and these aggregates to succeed; skipped/failed jobs or empty test groups do not
count as completed coverage. Per-job summaries identify package, Python, group,
tier and outcome; pytest also prints skip reasons. The final summary states the
selected tier. Repository protection settings are managed separately.

The current five packages use 17 jobs: 12 execution jobs, discovery, Python 3.6
compatibility, two legacy AI aggregates and final coverage. Each AI group installs its
local dependency closure and runs the package architecture/lint gates before
its own tests; files stay intact for fixture reuse and four-worker `loadfile` scheduling.
The file-family split preserves existing coverage and fixture locality, but is
not a guarantee of balanced wall time. Runner startup, installation and repeated
gates also affect both latency and total runner usage. Automatic execution jobs
have a 20-minute limit; explicit `all` runs allow 60 minutes for manual probes.

Named `manage.py install` selections include their transitive local dependencies
and requested extras in the same pip resolution. CI uses this for all packages
except core: core's CLI-help tests walk every installed package, so its jobs keep
installing all packages to retain that cross-package coverage. Editable installs
continue to include development dependencies; pip still enforces the original
version constraints and resolves external dependencies. The existing pip cache
and open dependency-update policy are unchanged.

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
