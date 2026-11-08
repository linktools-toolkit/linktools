# Task terminal write grouping

Task completion and failure remain synchronous durable boundaries. The node
transition, graph transition, result (on success), expansion records (when
present), and Task events are committed in one transaction. Lease/fence
validation, the graph event-owner guard, and unknown-commit readback are not
deferred or removed.

When a terminal node also changes graph status, their validated record
replacements share the existing `StateTransaction.replace_records` operation.
Each record retains its own identity and expected revision. A partial CAS match
aborts the entire storage group, including earlier result/expansion writes and
any provisional graph revision. The result and expansion INSERTs keep their
existing statement boundaries; combining them would increase a statement's
packet/parameter envelope without a demonstrated safe backend contract.

## Dialect scope

- SQLite and PostgreSQL use the existing multi-record CASE UPDATE path.
- MySQL keeps the existing row-wise `executemany` update path to preserve each
  value's packet boundary. Grouping the API call does not establish fewer
  server-side UPDATE statements or network round trips on MySQL.
- This change does not combine separate Task completions, reduce transaction
  counts, or introduce a timer/buffer for authoritative graph state.

## Matched SQLite measurements

The probe `tests/ai/test_task_terminal_write_batch.py` excludes provisioning,
admission, claim and final inspection. Before grouping, a single-node success
or failure used 3 UPDATE calls. After grouping, each uses 2 UPDATE calls.
Both versions use 3 SELECTs and one committed transaction. Successful completion
uses 3 INSERTs; failure uses 2. Dynamic expansion in the tested graph keeps
2 UPDATEs, 4 INSERTs, 5 SELECTs and one transaction, unchanged.

These are SQLAlchemy statement-execution counts, not physical disk IO or wire
bytes. PostgreSQL and real MySQL performance are not measured by this SQLite
probe. The row-wise path is exercised on SQLite for transaction/CAS semantics;
it is not a substitute for real MySQL acceptance.

Run the focused check through the repository entry point:

```sh
PYTHONPATH="$PWD:$PWD/linktools/src:$PWD/linktools-ai/src" \
PYTEST_ADDOPTS='-k "task_terminal_write_batch or task_sqlite_cas_convergence or sql_batch_write_failure" -s' \
python manage.py check linktools-ai --skip-compatibility
```
