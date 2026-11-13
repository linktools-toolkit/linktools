# Step effect ledger ownership and index cost

`ToolRepository` keeps its admission, fencing, generation, idempotency and recovery
contract. Canonical `tool_operation` keys now have one physical owner:
`ai_step_tool_operations`, not `ai_state_records`. Memory uses a separate owner
container under the same transaction overlay. Portable logical records remain
`StoredRecord`; import routes them back to their physical owner. External custom
StateStores retain their transaction interface and own their physical layout.
Opaque raw keys are still supported; a caller manually assigning a reserved kind
to an opaque key is not the built-in canonical ToolRepository path.

No trace adapter is required. An isolated Memory/SQLite diagnostic admitted and
settled a tool, read its alias, scanned it, rolled back a deletion, then deleted
it successfully. SQLite counted one effect row and zero generic tool rows.
A separate 140,432-byte Native receipt diagnostic preserved exact wrapper bytes:
warm read 2 SQL statements, reopened cold read 4; these are read costs, not a
claim about total execution cost. Formal regressions run in CI, not locally.

## SQL indexes

The table has 19 real columns and 6 indexes including the primary key:

| Index | Actual responsibility |
| --- | --- |
| primary `id` | Local surrogate used by Native owner edges |
| unique `store_key,key_locator` | Exact operation lookup, CAS and alias target validation |
| `scope_locator,subject_id` | Exact call lookup; its left prefix bounds run listings |
| `parent_locator,sort_prefix` | Execution-scoped effects/recovery listing |
| `updated_at` | Required audit index |
| `created_at` | Required audit index |

Constant `kind` and absent binary payload columns are projected, not persisted;
current writers validate these invariants before mutation. Different physical
tables use different Native edge kinds, so equal surrogate IDs cannot collide.

## Ordering-index comparison

SQLite isolated dataset: 4 tenant partitions × 128 runs × 128 calls = 65,536
rows, 4 runs per execution. Approximate VM instructions use a progress callback
at 100-instruction intervals. The candidate removes `(store_key,sort_prefix)`
and `(scope_locator,sort_prefix)`, retaining the six indexes above.

| Query | Rows | Original VM steps | Six-index VM steps |
| --- | ---: | ---: | ---: |
| Run listing | 128 | 7,600 | 6,500 |
| Execution listing | 512 | 30,700 | 30,700 |
| Exact scoped call | 1 | <100 | <100 |
| Unscoped partition sorted page | 128 | 7,300 | 180,500 |

EXPLAIN changes the run listing from scope/sort to the scope/subject index's
left prefix, then sorts only that run. Execution listing and exact calls retain
their indexes. Unscoped sorted paging instead scans that partition's 16,384 rows
through the unique index and sorts. Built-in ToolRepository listings use a run
or execution predicate; this tradeoff must not be generalized to workloads that
use raw unscoped pagination heavily. Snapshot/GC scanning orders by full key,
not the removed sort-prefix access path. These measurements are one workload,
not a universal performance claim. Ordinary payload/content sharing semantics
are unchanged by the whole-kind owner migration.

Reproduce with the project environment from the repository root:

```sh
PYTHONPATH=.:linktools-ai/src:linktools/src python tests/ai/_measure_effect_ledger_indexes.py
```

This standalone measurement uses a fresh in-memory SQLite database, not pytest.

## Whole-kind payload ownership

The subsequent first-writer change keeps all sizes and encodings of admitted
arguments, results and errors in the Step ledger itself. It creates no
`tool-operation-payload` content rows in the generic archive. The prior numeric
Native/snapshot diagnostics above predate this change and are not rerun evidence
for it. Initial large receipts make the Step ledger row wider, and identical
independent receipts are not deduplicated there. Fixed, exactly reconstructible
Native receipts still replace ledger bodies atomically after actual Step
publication. No threshold determines whether tool facts enter generic records.
This update is submitted for natural CI without running local tests or probes.

## Accepted execution payload content

`execution-payload:<digest>` immutable archive content now has the distinct
`ai_step_payload_contents` owner. Existing ContentArchive codecs and references
are unchanged. Binding and runtime-policy codecs remain with their original
shared owner. The deterministic complete key, not a mutable metadata registry,
selects the table for first write, CAS, direct/batch reads, specialized archive
frontier reads, scans, import and deletion. Memory uses the same disjoint owner
selection. The typed table has 13 physical columns and 4 indexes including its
primary key, unique `(store_key,key_locator)` and both required audit indexes.
Other invariant fields are validated then projected as constants.

This is first-pass implementation evidence only; no local verification was run
for the payload-owner change before submitting it to natural CI.

## Shared first-writer payload owner

The next first-pass change replaces inline tool bodies with the same canonical
`encode_domain(StoredPayload)` / `execution-payload` immutable Step content used
by accepted execution payloads and continuation arguments. This applies to every
size and encoding. Content and effect references are published within the same
storage-group transaction. Standalone tool mutations open the archive before
entering that transaction; an already-active transaction must have enlisted it.
Previously published exact Native receipts retain their fixed-source reference.
No new payload protocol or table is introduced. The preceding inline-owner
tradeoff describes the intermediate implementation, not this final writer.
This change has not been locally tested or probed before natural CI.
