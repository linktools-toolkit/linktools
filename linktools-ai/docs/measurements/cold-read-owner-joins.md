# Cold read owner/content joins

These are statement counts, not latency or physical disk I/O. The original PR
base is `bd5e38ab9da0f8b5c487917693c812723ecfb058`, not the intermediate
`6e907b06` revision.

## Matched workload

SQLite file storage with default configuration and no explicitly configured
application record cache. One session, two turns/executions, four offline
FunctionModel requests and two replay-safe tool calls each append one real file
effect. Small and large tool results are respectively 32 and 65536 bytes of
compressible text. Both sizes must retain exactly two effects through recovery
and offline GC; the measurements do not replay tools.

Cold readback closes Runtime, Native and Persistence, then reopens them. It makes
14 public calls: result, full History, tool execution-list, tool get and tool
agent-run-list for each of two executions, plus latest Native snapshot and agent
run get for each of two runs. Counts include SELECT and WITH in this phase only,
not reopening initialization, BEGIN/COMMIT, GC or measurement SQL. OS page caches
are not flushed. Warm readback has a different API set, so warm-to-cold totals
must not be interpreted as a cache penalty.

## Attribution before these two join changes

The same entry-attribution diagnostic on the original base and local `e04f07b4`
(published as `26708394`) passed full recovery/equality/no-replay assertions.
Each row aggregates two calls; both payload sizes have identical counts.

| API | Original base | e04 | Difference |
|---|---:|---:|---:|
| execution result | 6 | 8 | +2 |
| full History | 28 | 36 | +8 |
| tool execution-list | 2 | 6 | +4 |
| tool get | 2 | 4 | +2 |
| tool agent-run-list | 2 | 6 | +4 |
| latest Native snapshot | 4 | 4 | 0 |
| agent run get | 2 | 2 | 0 |
| Total | 46 | 66 | +20 |

## Two measured changes

1. Local `28e290e3` captures uncached SQL execution owners and their fixed
   contents through the existing JOIN in the same read snapshot. It replaces
   the preliminary observation plus separate owner/body/configuration reads.
   The Native range remains independently validated. Configured observation
   caches, Memory/custom stores and writable transactions retain their paths.
2. Local `338da395` reuses the existing scoped record selection inside the
   owner/content JOIN for tool lists. Store/scope/parent/kind filters, ordering
   and limit are shared with ordinary list reads. Owners without content are
   retained. The Native message query is not removed: it holds different data
   from the typed payload row.

| Phase | e04 | Execution join | Tool-list join |
|---|---:|---:|---:|
| Runtime SELECT + WITH | 363 | 363 | 363 |
| Warm readback | 57 | 51 | 49 |
| Cold readback | 66 | 60 | 56 |

Both sizes passed the same diagnostic workload at each candidate, including
complete results, History, tools, Native snapshots, offline-GC fixed point and
exactly-two-effects assertions. This is not a timing benchmark. The original
whole-storage measurement remains frozen at `c1132d01`; these read-only changes
do not relabel that earlier disk/row-byte evidence as a newly measured head.

The cold total is still above the original base's 46 reads. Fixed references
require authenticating actual owners and restoring independently held content;
that does not justify arbitrary extra round trips. In particular, result's
post-read queries were traced to terminal-retention and session confirmation,
not duplicate result hydration, and have not been deleted to improve a count.

Targeted cold-archive retry, missing-source rejection, owner deletion and cached
owner-race regressions accompany the execution change. Backend JOIN and scoped
selection regressions accompany the tool-list change. Exact-head natural CI is
required separately; local measurements are not a full-suite success claim.
