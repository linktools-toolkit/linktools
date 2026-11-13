# Transaction-local content decode reuse

Measured with the existing `test_projection_boundary_read_budget` workload,
Python 3.12, SQLite, 2 and 4 tool rounds. Baseline includes `bb2239a9` batch
configuration validation. Candidate replaces the binding-only transaction cache
with one inline archive decode cache; it does not add a second cache.

| Measurement | Before | After |
|---|---:|---:|
| Model-history payload decodes (both workloads) | 0 | 0 |
| Configuration/input/result-template decodes | 24 | 17 |
| Bytes passed through payload decompression/digest validation | 7207 | 5503 |
| Unique immutable owner identities | 4 | 4 |
| Two-round SQL reads (SELECT and WITH) | 191 | 191 |
| Two-round returned SQL rows | 188 | 188 |
| Returned text/binary value bytes | 216129 | 216129 |
| Other non-null scalar result cells | 961 | 961 |

Returned bytes count UTF-8 string values and binary values across the actual SQL
result rows, not driver/network framing, scalar storage sizes, database pages or
filesystem reads. Four rounds perform 299 SQL reads (172 record-family, 27
message-family, 100 other), versus 191 for two rounds (112/21/58). SQL did not
improve in this change. The improvement is repeated CPU decode/hash work.

The 17 remaining decodes are distributed across nine real transaction scopes:

| Scope, in capture order | Input | Binding | Policy | Result template |
|---|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 | 0 |
| 1 | 1 | 1 | 1 | 0 |
| 2 | 0 | 0 | 1 | 0 |
| 3 | 0 | 0 | 1 | 0 |
| 4 | 1 | 0 | 0 | 0 |
| 5 | 0 | 1 | 1 | 0 |
| 6 | 0 | 1 | 1 | 1 |
| 7 | 0 | 0 | 1 | 0 |
| 8 | 0 | 0 | 1 | 1 |

Each scope/identity now decodes once. These scopes include separate publication,
inspection, terminal publication and retention boundaries. Reusing values across
these transactions would discard required current-source validation.

The regression keeps the original model-body budgets (`6 * rounds + 5` for
projection and `3 * rounds + 2` elsewhere). It separately asserts owner costs:
no repeated decode per transaction/identity, at most four owner identities,
17 decodes and 5503 decoded bytes for both measured workloads. It does not rename
24 decodes as 24 disk reads, relax the old model-body budget, or move any body
back into execution records.

Integrity checks cover same-transaction corruption and deletion, fresh validation
in a later transaction, and rejection of boolean/float sizes. Source metadata is
still captured before reuse. Only inline bytes are reused; object-file bodies
are always read and verified. Numeric comparison is safe because `ContentRecord`
rejects bool/float sizes and requires exact bytes for inline bodies, with no JSON
mapping in the reusable representation.
