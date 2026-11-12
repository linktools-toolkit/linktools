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
