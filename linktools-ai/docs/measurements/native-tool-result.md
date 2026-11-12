# Native tool receipt projection

The 2026-11-12 isolated SQLite probe uses the real terminal StateCommands
publication, one settled tool operation, and a 130,432-byte ordinary text receipt.
The original effect is committed before its Step message exists. Publication
then replaces the result slot only after the immutable Native source is written,
inside the same group transaction. It retains the original runtime/operation ID,
timestamp and other wrapper fields; restored receipt bytes match exactly.

Reproduce with the project dependencies available:

```
PYTHONPATH=.:linktools-ai/src:linktools/src python -m tests.ai._measure_native_tool_result
```

This standalone measurement does not run pytest or dispatch CI.

| Operation | SELECT | WITH range query | BEGIN |
| --- | ---: | ---: | ---: |
| Warm single tool read | 1 | 1 | 1 |
| Genuine reopened cold single read | 3 | 1 | 3 |
| Warm one-call batch read | 1 | 1 | 1 |

`WITH` is a real SQL query, not excluded from the read budget. Warm reads therefore
use two queries, and cold discovery/retry uses four. Whole terminal publication
uses 15 SELECT, one WITH, 20 INSERT, 10 UPDATE, four DELETE and three BEGIN. These
are whole-operation totals, **not** the incremental cost of sharing.

Original receipt: 130,432 bytes. Retained template: 434 UTF-8 bytes. Complete tool
owner JSON after sharing: 1,835 bytes. These figures are not a total database size
claim. Before sharing the large receipt already has a small ContentArchive
reference. Replacing that reference can increase the owner row while removing
one live dependency on the duplicate archive body. The old compressed archive
body remains until existing shared-content GC determines that no owner retains
it. There is no per-tool full-table scan or direct shared-body deletion. Physical
SQLite file size also need not shrink when logical rows are collected.

Candidate roots come only from newly captured checkpoint/observation chunks,
not the full model-context projection. Small or absent new tool text performs
no added query. Large-candidate publication reads actual fixed source bytes,
reads the corresponding scoped operation records, then changes the owner via
CAS and existing Native owner-edge maintenance. Initial effect settlement is
unchanged. Source content is never moved into ObjectStore.

Coverage is deliberately precise: large ordinary string ToolReturnPart results
with canonical base64 receipt bytes and exactly recoverable wrappers. Small,
opaque, noncanonical, attachment-bearing, structured/complex, different-content,
custom-repository and split-group cases retain their original receipt path.
Recovery aliases from a different agent-run owner are not rewritten by this
first implementation. Failure/error and argument payloads are unchanged.

Independent probes additionally exercised a real non-terminal public
sync_projection and an injected failure *after* the Native-ref owner replacement.
The latter rolled back the reference and terminal publication while preserving
the previously settled original effect and STARTED execution. CI regressions
cover direct/prepared boundaries, equal/unequal/small cases, cold recovery,
noncanonical/opaque fallback and rollback. Formal suites were not run locally.

## Snapshot and collection probe

A separate end-to-end SQLite probe used a 140,432-byte original receipt (a
slightly different repeated text fixture from the SQL probe above):

- Immediately after conversion, the old tool ContentArchive body still occupied
  one row with 1,245 compressed `payload_bytes` bytes, excluding row metadata.
- After the existing shared-content GC ran, that old archive body occupied zero
  rows. The 1,245-byte measurement is compressed physical payload, not the
  140,432-byte original logical receipt size.
- Read-only export produced an 18,869-byte snapshot manifest. Restore preserved
  the exact original tool receipt bytes.
- Removing other transcript/record owners and collecting left one Native message
  alive solely for the tool operation, and the tool receipt still read exactly.
- Removing the final tool owner and collecting left zero Native messages and
  zero Native sequences.

The probe does not claim immediate deletion during conversion, immediate SQLite
file shrinkage, or a general compression ratio. The formal CI regression is
`tests/ai/test_native_tool_portable_gc.py`; it covers the same ownership and
collection transitions without asserting environment-dependent byte counts.
