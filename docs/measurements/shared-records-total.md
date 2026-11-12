# Matched whole-records SQLite measurement

## Scope and result

The unchanged fixed workload reduced whole `ai_state_records` column-value bytes from **195,433 to 141,330 bytes (27.6837%)**. This includes every column, generated values, binary payloads and all archive rows. It is not a production capacity or speed claim, and it must not be added to percentages from Tool, Rule or Native-output-specific workloads.

**Coverage limit:** 20 Agent executions, 20 Task executions and 20 graph binding captures. No Tool requests/receipts, retained Rule contexts, graph admission storage or large Native/Step terminal output bodies are populated. Only SQLite was measured.

## Exact sources and reproducibility

| Item | Baseline | Measured candidate |
| --- | --- | --- |
| Commit | `c99c687d7b325d2b34350080c76bae5023324313` | `cdbdce17e2ccfe0715a73e6873c702a9cec98575` |
| Tree | `4990e07e00e45438e718a01db549ff23cba7f250` | `43167e5098a4306af7f77bc69df7a99ddd30e98c` |

The candidate is an intermediate integration revision, not a claim about an unmeasured later head. The documentation commit does not change the measured source tree.

- Script: `tests/ai/_measure_shared_records.py`, unchanged in both runs.
- Script SHA-256: `087f9bfb1044b458cea5f496cf76bc607d9698b2ed165ffe1a7f56ef4737ffce`.
- Input SHA-256: `db06aa6d773aad468007229c3e347f67fdf5ebb7100b41153a5e5aebd99b54a7`.
- Interpreter: `3.12.14 (main, Aug 25 2026, 14:00:49) [Clang 22.1.3 ]`.
- Both runs used fresh databases, separate processes, the same interpreter, and identical full installed dependency name/version mappings. Imports were verified to resolve to each requested checkout rather than the installed editable source.
- No models or tools executed. No pytest, formal suite or CI was run for this diagnostic.

Example invocation, repeated with a separate database/output for each checkout:

```sh
PYTHONDONTWRITEBYTECODE=1 python tests/ai/_measure_shared_records.py \
  --source-root "$SOURCE_ROOT" --expected-sha "$EXPECTED_SHA" \
  --database "$DATABASE" --output "$OUTPUT" --label "$LABEL"
```

## Byte and physical-storage definitions

`column_value_bytes` is the sum of `length(CAST(value AS BLOB))`, treating NULL as zero, for every SQL column including generated columns. Numeric and date values use their SQLite textual representations. This is a logical column-value sum, not SQLite physical encoding or unique information size.

`payload_bytes` is included: baseline **0 bytes**, candidate **4,020 bytes**, in archive content. Any earlier diagnostic note saying the candidate binary column was empty was incorrect; the reported whole-table total already included those bytes.

Physical B-tree measurements come separately from `dbstat`, including the records table and all its indexes. No VACUUM or manual checkpoint was used. File sizes are sampled after closing the cold-read connection; WAL and SHM were both zero.

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Records rows | 61 | 64 |
| All records column-value bytes | 195,433 | 141,330 |
| JSON payload column bytes | 170,901 | 111,116 |
| Binary payload column bytes | 0 | 4,020 |
| Records plus index pages | 77 | 67 |
| Records plus index page bytes | 315,392 | 274,432 |
| Records plus index physical payload bytes | 223,728 | 171,422 |
| Records plus index unused bytes | 88,986 | 100,404 |
| Database file bytes | 610,304 | 569,344 |
| WAL bytes | 0 | 0 |
| SHM bytes | 0 | 0 |

## Whole-records breakdown by kind

| Kind | Baseline rows | Baseline bytes | Candidate rows | Candidate bytes |
| --- | ---: | ---: | ---: | ---: |
| `archive_content` | 0 | 0 | 3 | 6,216 |
| `archive_scope` | 1 | 281 | 1 | 281 |
| `execution` | 40 | 150,012 | 40 | 109,473 |
| `task_graph_binding_capture` | 20 | 45,140 | 20 | 25,360 |

All other business tables were empty in these runs. Archive storage was not excluded or treated as free.

## SQL and recovery evidence

SQLAlchemy event counters include initialization, workload, reopen and cleanup. Measurement queries use a separate sqlite3 connection and are excluded. DML counts executed INSERT/UPDATE/DELETE/REPLACE statement calls; commits are transaction commit events, not elapsed-time or durability throughput measurements.

| Counter | Baseline | Candidate |
| --- | ---: | ---: |
| statement_calls | 973 | 1202 |
| SELECT | 362 | 487 |
| DML | 141 | 201 |
| commit_events | 381 | 425 |
| rollback_events | 2 | 2 |

Each execution had two CAS updates, two warm views, two cold views and one cold full read. Both processes completed the helper assertions for create equality, CAS equality, view kind/event sequence, full-record equality after reopen, and capture instructions after cold restore. Cold means a new Persistence/repository instance, not a cleared operating-system cache. This does not establish cross-version database migration compatibility.

## Earlier stages and superseded counters

| Measured stage | Records bytes | SELECT | DML | Commits | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `d2e4e57d710a66700d9b8644fff418a3aecaaede` | 141330 | 487 | 201 | 425 | Earlier stage |
| `e91adab9ad2002f39a6682a91236d6613c8b9ff9` | 141330 | 488 | 201 | 385 | Superseded Native stage with get/get_fresh cache-contract regression |
| `cdbdce17e2ccfe0715a73e6873c702a9cec98575` | 141330 | 487 | 201 | 425 | Latest measured candidate, cache contract restored |

**The old 385-commit number is not applicable to the latest measured candidate.** It must not be presented as the current optimization result. All stages above used the same fixed workload; the latest candidate was actually rerun rather than inheriting earlier counters.

## Full matched dependency manifest

The full name/version mappings were compared for equality before summarization. Both runs reported:

```json
{
  "Authlib": "1.9.5",
  "PyJWT": "2.15.1",
  "PyYAML": "6.0.3",
  "Pygments": "2.21.0",
  "SQLAlchemy": "2.0.51",
  "aiofile": "3.12.3",
  "aiosqlite": "0.22.1",
  "annotated-types": "0.8.0",
  "anthropic": "1.17.0",
  "anyio": "4.16.0",
  "asyncmy": "0.2.18",
  "asyncpg": "0.32.0",
  "attrs": "26.1.0",
  "beartype": "0.23.1",
  "cachetools": "8.0.0",
  "caio": "0.12.10",
  "certifi": "2026.10.23",
  "cffi": "2.1.1",
  "charset-normalizer": "3.5.2",
  "click": "8.5.1",
  "cryptography": "50.0.2",
  "dnspython": "2.9.2",
  "docstring_parser": "0.18.0",
  "email-validator": "2.3.0",
  "exceptiongroup": "1.3.1",
  "fastmcp-slim": "4.1.8",
  "filelock": "4.2.10",
  "genai-prices": "0.2.3",
  "greenlet": "3.5.6",
  "griffelib": "2.4.1",
  "h11": "0.16.0",
  "httpcore": "1.0.9",
  "httpcore2": "2.16.0",
  "httpx": "0.28.1",
  "httpx2": "2.16.0",
  "idna": "3.20",
  "iniconfig": "2.3.1",
  "jiter": "0.19.0",
  "joserfc": "1.8.0",
  "jsonschema": "4.26.0",
  "jsonschema-specifications": "2025.9.1",
  "linktools": "0.10.0",
  "linktools-ai": "0.10.0",
  "logfire-api": "5.6.0",
  "markdown-it-py": "4.3.0",
  "mcp": "2.3.0",
  "mcp-types": "2.3.0",
  "mdurl": "0.1.2",
  "openai": "3.49.0",
  "opentelemetry-api": "1.45.2",
  "packaging": "26.3",
  "pip": "25.0.1",
  "platformdirs": "4.14.5",
  "pluggy": "1.7.0",
  "psutil": "7.2.2",
  "py-key-value-aio": "0.4.6",
  "pycparser": "3.12",
  "pydantic": "2.14.3",
  "pydantic-ai-harness": "0.68.0",
  "pydantic-ai-slim": "2.68.0",
  "pydantic-graph": "2.68.0",
  "pydantic-settings": "2.16.2",
  "pydantic_core": "2.50.4",
  "pytest": "9.2.0",
  "pytest-asyncio": "1.4.1.post1",
  "python-dotenv": "1.2.4",
  "python-magic": "0.4.27",
  "python-multipart": "0.0.33",
  "referencing": "0.37.0",
  "regex": "2026.11.9",
  "requests": "2.34.2",
  "rich": "15.0.0",
  "rpds-py": "2026.9.1",
  "sniffio": "1.3.1",
  "sse-starlette": "3.7.0",
  "starlette": "1.10.0",
  "tiktoken": "0.14.0",
  "tomlkit": "0.15.2",
  "truststore": "0.10.5",
  "typing-inspection": "0.4.5",
  "typing_extensions": "4.16.0",
  "urllib3": "2.8.0",
  "uvicorn": "0.54.0"
}
```
