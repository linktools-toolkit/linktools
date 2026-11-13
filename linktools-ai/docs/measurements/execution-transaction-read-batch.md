# Execution transaction content reads

Matched local measurement, 2026-11-13, base `c2566e290b61019043da652a4d04a4aa819f11af`.
This is a narrow optimization, not completion of the overall IO work.

## Scenario

The existing `test_official_runtime_attribution.py` fixture: one session, two turns,
eight model calls and six tool effects, four-byte tool results; file-backed SQLite,
InMemoryCacheBackend enabled, TraceCapability disabled, no budget or coordination.
Checkpointing and frontier capture are enabled. Counts cover each session start through
wait and cleanup; initialization, later history inspection and replay assertions are
outside the counting window. This is **not comparable** to historical 244/318-read
samples with other configurations.

## Change

`ExecutionRepositoryImpl.get_in_transaction` uses the existing content-owner batch
reader, as its observed-read counterpart already did. Previously terminal validation
loaded input payloads and binding/policy separately. Full decoding, source validation,
authorization and final CAS remain unchanged. No new cache or schema is introduced.

| Metric, both turns | Before | After |
| --- | ---: | ---: |
| SELECT | 79 (38 + 41) | 77 (37 + 40) |
| INSERT | 131 | 131 |
| UPDATE | 46 | 46 |
| Transactions / commits | 51 / 51 | 51 / 51 |
| Write commits | 38 | 38 |
| Bound text/blob bytes | 222,968 | 224,892 |
| Decoded State-record JSON bytes | 31,562 | 38,445 |

The joined read repeats the small execution owner alongside its fixed content, so
it trades two round trips for additional row/parameter bytes. These are not total
wire bytes or a measured latency improvement. No writes or commits were hidden or
moved outside the measurement window. Two initialization PRAGMAs observed during
these turns are counted separately and unchanged.

## Validation

The matched fixture passed before and after. The content-source join and execution
owner-write regression files passed: 29 passed, 2 skipped. A focused regression
checks complete transactional reconstruction and reuse within that transaction.
Tests use the repository's `manage._run_pytest` entry point; no CI dispatch or rerun
was performed. MySQL/PostgreSQL service validation remains with the existing owner.

## Follow-up: typed payload publication by content address

Typed `execution-payload` bodies now use insert-if-absent in their existing Step
payload partition instead of reading/decompressing existing bytes before publishing.
The first stored representation wins, including when equivalent content has raw and
zlib representations. This removes redundant content-address body comparison, not
owner identity, tenant isolation, source decoding, CAS or transaction rollback.
Other immutable State records and Native messages/sequences retain their contracts.

The SQL transaction caches a candidate revision only when insertion is confirmed.
Duplicate/unknown outcomes do not overwrite the winner's representation or GC
revision. A mixed insert still invalidates the partition's read cache. Memory uses
the same archive-partition semantics. No table, cache or recovery mechanism is added.

Same fixture and counting window:

| Metric, both turns | Prior batch-read change | With payload publication change |
| --- | ---: | ---: |
| SELECT | 77 (37 + 40) | 63 (30 + 33) |
| INSERT | 131 | 131 |
| UPDATE | 46 | 46 |
| Transactions / commits | 51 / 51 | 49 / 49 |
| Write commits | 38 | 38 |
| Bound text/blob bytes | 224,892 | 224,291 |
| Decoded State-record JSON bytes | 38,445 | 47,214 |

Decoded State-record bytes are a partial instrumentation counter, not total returned
SQL bytes. They rise while SELECT count falls, so this result does not claim every
IO metric improved or that overall IO work is finished.

Validation after final edits: 60 relevant regression/probe cases passed; the five
new focused payload cases passed separately. They cover raw/zlib winner reuse,
nonzero GC revision, a guard before duplicate publication, concurrent duplicates,
no existence pre-read, mixed old/new insertion with cached absence, and unknown
insertion counts. Existing tool tests cover owner rollback and portable cold recovery.
