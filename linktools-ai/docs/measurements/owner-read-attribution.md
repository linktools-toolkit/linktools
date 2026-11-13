# Fixed owner read attribution

Same v3 workload and installed dependencies, SQLite, two executions and exactly two
side effects. All warm/cold result, history, tool, Native snapshot, agent-run and
post-offline-GC equality assertions pass. Reads count both SELECT and WITH; cold
means reopening, not eviction of OS/database page caches. These are statement
counts, not disk reads or latency measurements.

| Source | Runtime small / large | Warm small / large | Cold small / large |
| --- | ---: | ---: | ---: |
| Original `6e907b06` | 324 / 326 | 45 / 49 | 48 / 56 |
| Whole-owner `b436644a` | 367 / 367 | 61 / 61 | 72 / 72 |
| Narrow tool metadata + control CAS `0dc74631` | 365 / 365 | 57 / 57 | 68 / 68 |
| Native-only equality `1d198b8e` | 363 / 363 | 57 / 57 | 68 / 68 |
| Keyed tool/content JOIN `db934ace` | 363 / 363 | 57 / 57 | 66 / 66 |

The last three rows and original were rerun with the same observational SQL
attribution script. The b436 row is the preceding matched v3 capture. The diagnostic
adds no queries, asserts that event and original statement totals agree, and does
not modify the original v3 script. File/stack instrumentation timings are not
performance evidence.

## Verified reductions

- History needs only canonical tool identity, execution membership, model call ID
  and status. Its narrow typed read retains owner validation while avoiding unused
  arguments/receipt hydration: four fewer warm and cold reads per workload.
- A control CAS retaining byte-identical content-addressed terminal facts need not
  consume their bodies. Effective policy equal to the original content identity
  needs no baseline hydration: two fewer runtime reads. A missing source is not
  recreated; a later actual consumer still rejects it.
- Native equality checks consume only Native values. Skipping unrelated ordinary
  input/result/recovery payload hydration saves another two runtime reads. Fixed
  root/scope/part and source-byte integrity checks remain mandatory.

## Remaining cold cost before keyed JOIN (small case, 68 vs 48)

Exact API labels, not attribution inferred from table row counts:

- Tool execution-list, individual get and run-list: +4 each, total +12. Each of two
  operations reads a typed argument owner and its fixed Native receipt source.
- Execution result: +2 for the two fixed Native outputs.
- History: +6, comprising two typed inputs, two Native outputs and two canonical
  tool metadata reads.

Large original bodies already required eight more archive reads, hence its net
cold increase is twelve rather than twenty. Reading independently owned content
is necessary; this does **not** prove each currently separate SQL round trip is
necessary. Keyed owner/content JOIN reuse removes two cold individual-get round trips,
bringing the small net increase to eighteen and large to ten. Warm/runtime
counts do not change. List queries retain their existing scoped selection and
Native receipts retain their recursive integrity/attachment path. No ordinary
SHA JOIN is substituted for that Native validation.
Runtime still exceeds the original by 39/37 reads; this report does not claim the
whole IO target is complete. Raw SQL table buckets alone are insufficient to
assign those remaining reads to callers.

## Regression boundaries

Control CAS keeps large-body re-encoding/read prohibitions, immutable terminal
conflict checks, and missing-source consumer failure on Memory and both SQLite
configurations. Native-only equality has a real terminal writer regression: no
ordinary content reads, unchanged fixed Native reference, and corrupted Native
bytes rejected in the next transaction. Offline ownership tests retain Native
body/encoding/cold-range bounds and prohibit online edge/collector DML. Typed
payload rows in `ai_step_messages` are counted separately by exact store partition,
not mistaken for duplicated Native messages.

The keyed JOIN regression performs a genuinely cold first get without opening
the archive beforehand. When discovery is necessary, the initial owner read is
discarded and a new snapshot captures the current owner with its fixed sources.
Subsequent duplicate/missing-ID batches require one ordinary-owner/content query
plus one Native range query when applicable; wrong-tenant and empty batches read
nothing. Deleting the actual typed or Native source still yields an integrity
error. Memory/custom/writable paths use their existing fallback.
