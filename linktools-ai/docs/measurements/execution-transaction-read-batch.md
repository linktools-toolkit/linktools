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
