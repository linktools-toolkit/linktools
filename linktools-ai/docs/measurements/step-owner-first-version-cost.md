# First-version Step ownership: matched end-to-end cost

This is a diagnostic of the complete first version, **not evidence of a net space or I/O improvement**.

- Baseline: `6e907b06d6d4342a51c0d31f6fcedaa9a6cad671`.
- Candidate: `3596f28238164fe4a7a26277b5e674abf8c4b1b1`.
- Same `measure.py`, `storage_helper.py`, input digest, Python and dependency inventory, verified equal in both result files.
- Two real Runtime/FunctionModel turns, one replay-safe tool effect per turn; small tool result 32 bytes and large result 65,536 bytes. Exactly two effects and four model requests per case.
- Warm/cold execution result, history, Native snapshots, agent runs and tool list/get/batch APIs agree. Cold means reopening, not flushing OS caches.
- No test runner, model service, VACUUM, manual database checkpoint or source modification was used.

All SQLite `ai_*` tables are enumerated, including new Step owner tables. “Column bytes” sums `length(CAST(column AS BLOB))` over **all columns**; numeric/time values use SQLite text conversion, so this is not physical storage. Database and B-tree page sizes are separately reported. SQL measurement uses SQLAlchemy events; independent read-only measurement queries are excluded.

## Space after close

| Metric | Small baseline | Small candidate | Large baseline | Large candidate |
|---|---:|---:|---:|---:|
| Runtime records, all-column bytes | 61,117 | 51,335 | 63,218 | 51,344 |
| All Step tables, all-column bytes | 28,055 | 50,031 | 27,877 | 49,842 |
| Records + all Step columns | 89,172 | 101,366 | 91,095 | 101,186 |
| All business table columns | 142,632 | 154,925 | 144,571 | 154,761 |
| Main database bytes | 528,384 | 593,920 | 520,192 | 593,920 |
| Database pages (4 KiB) | 129 | 145 | 127 | 145 |
| Records + Step B-tree bytes, including indexes | 262,144 | 323,584 | 253,952 | 323,584 |

WAL and SHM sizes after close are zero, and each database has one free page. All-business column bytes increased **8.62% / 7.05%**; main-file size increased **12.40% / 14.17%**. The 16.01% / 18.78% records reduction is therefore **not** net savings.

`agent_run` and `tool_operation` are absent from Runtime records. Candidate physical owners contain four run admissions (4,322 column bytes), two tool operations (5,576 bytes), and eleven payload contents (12,078 / 12,068 bytes). Runtime `execution` itself increases from 11,071 to 11,381 bytes. Other dominant unchanged records include model interactions (~18.9 KiB) and context projections (~4.1 KiB).

## I/O

Reads count both SELECT and WITH. DML counts statement calls, not affected rows.

| Phase / metric | Small baseline | Small candidate | Large baseline | Large candidate |
|---|---:|---:|---:|---:|
| Two-turn reads | 324 | 415 | 326 | 415 |
| Two-turn DML | 445 | 469 | 447 | 469 |
| Two-turn commits | 144 | 144 | 144 | 144 |
| Warm reads | 45 | 65 | 49 | 65 |
| Cold reads | 48 | 72 | 56 | 72 |
| Entire lifecycle reads | 428 | 566 | 442 | 566 |
| Entire lifecycle DML | 446 | 470 | 448 | 470 |
| Entire lifecycle commits | 217 | 220 | 217 | 220 |

Runtime reads increase **28.09% / 27.30%**. Cold reads increase **50% / 28.57%**. Runtime commits do not increase; the three additional lifecycle commits belong to setup/reopening.

A second matched run added read-only SQL tracing to the same workload on both revisions. Small-case runtime query-family deltas account for all 91 additional reads:

- Runtime-records-only queries: +30.
- Step-payload-only queries: +26.
- Native sequence-only queries: +7.
- Native message/sequence range queries: +20.
- Mixed records/run-admission queries: +6.
- Mixed records/payload queries: +2.

Query families identify tables, **not a proven call-stack attribution**. Binding/policy owner expansion and Native input candidate selection are candidates for focused attribution, rather than claims established by this count alone.

## Concrete overhead and optimization candidates

Small-case payload owners total 12,078 bytes: only 4,145 encoded body bytes, versus **7,933 bytes of other columns (~65.7%)**. Each row has roughly 721 bytes of identity, lookup, metadata and audit values before its body.

- Empty terminal diagnostics (`{}` and `None`) consume one 998-byte owner: 277 body bytes + 721 overhead. The two execution references consume 600 logical JSON value bytes, versus 38 bytes for the prior empty values. Omitting a schema-defined empty/default owner can eliminate this without duplicating a nonempty fact or reintroducing a size threshold.
- Two result metadata templates consume 2,168 bytes, with only 725 body bytes. Input owners consume 1,995 bytes. The two argument owners consume 1,846 bytes for two eleven-byte tool arguments. Typed wrapper and physical owner overhead matter even though ownership is unique.
- The new fixed Runtime policy owner consumes 909 bytes (184 body + 725 other columns). Records lose inline binding summaries, but add fixed policy references and policy reads. Batch/cache only within the existing verified transaction/source boundaries; do not restore duplicated summaries as a shortcut.
- Two input payload rows remain in the pre-GC inventory. These rows alone do not establish that executions still reference them. Subsequent call-stack instrumentation confirmed successful Native input projection; normal GC reclaimed the obsolete initial owners. A negative-candidate prefilter showed no read reduction for this workload and was not integrated.
- Simplifying physical payload identity/lookup representation may reduce fixed overhead, but this requires a real typed-table/query design and renewed DDL review. Do not delete required audit indexes, add padding columns, or call relocation itself a storage optimization.

The large body is repetitive and compresses strongly: its archived receipt body is roughly 0.9 KiB, not 64 KiB on disk. These two examples do not establish a general large-content compression result. No latency or non-SQLite performance claim is made.

This measurement precedes subsequent fixes and optimizations. Rerun the same workload and full accounting against the final candidate before making improvement claims.


## Follow-up: verified call stacks, normal GC and narrow read reuse

The same workload and dependencies were rerun at `fc938608` against baseline
`6e907b06`. SQL attribution used transaction-local context tags around Native
root capture, because SQLAlchemy's greenlet stack alone loses the async caller.
Across both cases, current-only captures included four successful input projections,
16 `get_observed` reads from Runtime persistence, four each from terminal effective
commit, terminal decode, terminal encode and postcommit observation, four retention
encode reads, and eight history execution reads. These are call-stack observations,
not the earlier table-family hypothesis of failed candidate discovery.

Both sides then ran the same public `purge_unreferenced_objects` under offline
exclusivity after closing Runtime, followed by reopening and exact result, history,
Native snapshot, run and tool receipt checks. No owner was deliberately released.
No vacuum was run.

| Case / revision | All business column bytes before / after GC | DB bytes before / after GC |
|---|---:|---:|
| Small baseline | 142,632 / 142,632 | 528,384 / 528,384 |
| Small fc938608 | 153,290 / 151,295 | 589,824 / 589,824 |
| Large baseline | 144,562 / 144,562 | 520,192 / 520,192 |
| Large fc938608 | 153,128 / 151,133 | 589,824 / 589,824 |

Candidate GC removed two obsolete initial input payload rows, 1,995 column bytes,
leaving eight payload owners. Even after symmetric GC, total candidate business
bytes remain higher: approximately 6.1% small and 4.5% large. Allocated pages do
not shrink. This is not a net space-saving claim.

A narrow follow-up at `70c10f72` reuses only the original value just decoded in the
terminal writer's same-transaction, no-source-write window. The original stored
projection must match; candidate values are not historical proof. It adds no
cross-transaction cache and does not skip source validation after publication.
On each two-turn case, runtime SELECTs fell from 402 to 400; WITH stayed 1, DML
469, commits 144. Warm/cold reads were unchanged. All workload semantic checks
passed. Separate bounded probes covered direct/prepared input reuse, publication
rollback, and eight unknown-commit/corruption outcomes. Formal suites remain CI's
responsibility.
