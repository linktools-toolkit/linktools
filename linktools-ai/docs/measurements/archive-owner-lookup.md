# Archive owner lookup

Direct content addresses are now selected from their key-determined physical
owner using `(store_key, key_locator)` plus exact raw key, scope, and kind
checks. They no longer pass through the content sort-identity join. Frontier
metadata-only queries select only the sequence table. Mixed requests remain one
statement and one database snapshot, with duplicate results removed by UNION.

A standalone SQLite diagnostic with real published archive content measured:

- Direct execution-payload: one SELECT; EXPLAIN reports
  `SEARCH ai_step_payload_contents USING INDEX ... (store_key=? AND key_locator=?)`.
  No generic records table occurs in this SQL.
- Sequence metadata-only: one SELECT; no Step payload table occurs in its SQL.
- Compilation of 500 all-slot node requests plus 500 direct keys, using existing
  batching: maximum binds SQLite 947, MySQL 999, PostgreSQL 914.

The paired CI regression is
`test_direct_step_content_uses_exact_index_without_sequence_union` in
`tests/ai/test_archive_frontier_storage.py`. Formal tests were not run locally.
No DDL, index, parameter budget, byte validation, or scope contract changed.

Indirect sequence body reads still permit every supported archive codec and
therefore retain both possible content owners. A sequence key alone does not
identify the content codec; excluding the Step owner there would break valid
execution-payload sequences. These measurements do not claim a reduction in
whole-runtime SELECT count or storage size.

## Indirect frontier owner joins

Indirect body lookups now join each physical owner to the bounded identity set
before combining the results. This avoids materializing the entire normalized
owner union. The Step branch is gated on the exact `execution-payload:` identity
prefix; other content codecs cannot visit its rows. LEFT JOIN keeps the small
identity set as the driving relation, including on an unanalyzed SQLite store.
A real SQLite archive diagnostic reports the generic target as
`SEARCH ... (store_key=? AND kind=? AND sort_prefix=?) LEFT-JOIN`, with no scan of
the physical generic table.

MySQL 5.7 has no CTE support, so the bounded identity subquery is repeated for
the two possible owners. Batch sizing accounts for those actual repeated binds
without increasing the 999-bind ceiling: the same 500-node/500-key compilation
probe reached SQLite 894, MySQL 998, PostgreSQL 712 binds. This may split very
large indirect requests into more statements; it is not a whole-runtime I/O
improvement claim. Direct-key and metadata-only requests retain one owner plan.

Known-new tool settlement now checks one immutable Step content batch even when
its effect owner needs no read. The content can be shared with another effect,
so a new operation is not proof that its content is absent. The test preserves
zero effect/alias SELECTs while allowing at most one exact content-key batch;
existing-content byte integrity checks are retained.
