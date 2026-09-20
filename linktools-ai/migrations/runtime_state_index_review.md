# Runtime State Index Review

This note defines the DBA review boundary for the five Runtime StateStore tables.
`init_schema.sql` and `runtime/state/_schema.py` remain the schema authorities.

## Target index contract

| Table | Columns | Unique | Business | Audit |
| --- | ---: | ---: | ---: | ---: |
| `ai_state_records` | 15 | 1 | 2 | 2 |
| `ai_state_aliases` | 6 | 1 | 1 | 2 |
| `ai_state_facts` | 11 | 1 | 2 | 2 |
| `ai_state_sequences` | 6 | 1 | 0 | 2 |
| `ai_state_operations` | 10 | 2 | 1 | 2 |

Every table also has its surrogate primary key. `ix_updated_at` and
`ix_created_at` are required by `linktools-ai/AGENTS.md` and are not
candidates for removal.

The retained business indexes map to real access paths:

- records: kind-ordered pagination and scope-ordered pagination; state and
  parent predicates are residual filters, while maintenance scans reuse the
  store-scoped unique key;
- aliases: reverse lookup by `record_key_digest` during record deletion;
- facts: owner cleanup and stream+subject ordered lookup;
- operations: stream+state ordered lookup for pending/running and terminal
  compaction queries.

The following ordinary indexes are not part of the target contract and must not
be recreated: `ix_scope_digest_state_sort_key`,
`ix_parent_digest_sort_key`, `ix_store_digest_kind_key_digest`,
`ix_store_digest_alias_digest`, `ix_store_digest_stream_digest_sequence`,
and `ix_store_digest_key_digest` on sequences or operations.

## Existing database review

Before changing an existing MySQL database, capture the actual table definition
and index list for all five tables:

```sql
SHOW CREATE TABLE ai_state_records;
SHOW CREATE TABLE ai_state_aliases;
SHOW CREATE TABLE ai_state_facts;
SHOW CREATE TABLE ai_state_sequences;
SHOW CREATE TABLE ai_state_operations;

SHOW INDEX FROM ai_state_records;
SHOW INDEX FROM ai_state_aliases;
SHOW INDEX FROM ai_state_facts;
SHOW INDEX FROM ai_state_sequences;
SHOW INDEX FROM ai_state_operations;
```

Count distinct index names, not one row per indexed column. Compare complete
ordered column lists, uniqueness, prefix length, and collation before deciding
that two indexes are equivalent.

If an old duplicate index or old global unique key exists, generate deployment
DDL only after confirming the replacement store-scoped unique key exists and
that its target columns contain no duplicates. Do not ship a blind drop script
for an unknown database state.

For `ai_state_records`, the target contract intentionally caps secondary
MySQL indexes at five for fifteen columns, satisfying the one-third index-count
rule without dummy columns or removing the required audit indexes. The schema
contract test enforces this budget.
