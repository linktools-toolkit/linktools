# Tool content database references

Manual SQLite probe, 2026-11-12. Baseline eac69900; candidate 7ab90dd2 plus the
snapshot restore closure correction. No schema changes. Formal test suites were
not run locally; regression coverage is intended for CI.

Reproduce with the same installed project dependencies and isolated checkouts:

```sh
python tests/ai/_measure_tool_content.py /path/to/baseline
python tests/ai/_measure_tool_content.py /path/to/candidate
```

Fixture: seed 13, 13 distinct alphanumeric byte payloads: one 369816-byte body and
12 5500-byte bodies. This approximates the reported size distribution, **not the
user's actual contents or compression ratio**. Existing 64 KiB inline policy is
unchanged. Only the large payload is archived. No ordinary text enters ObjectStore.

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| tool_operation JSON, 13 records | 596026 B | 103018 B |
| archive_content, count | 0 | 1 |
| archive_content JSON + binary | 0 B | 352526 B |
| archive_scope JSON | 61 B | 61 B |
| total record JSON + binary | 596087 B | 455605 B |
| SQLite file size | 942080 B | 798720 B |
| complete SELECT, 13 operations | 13 | 14 |
| complete UPDATE | 13 | 13 |
| complete INSERT | 0 | 1 |
| batch full read SELECT, 13 operations | 1 | 2 |
| admit SELECT / INSERT | 13 / 13 | 13 / 13 |

Logical bytes count only payload_json and payload_bytes, not keys, indexes or
all SQLite overhead. File bytes are an observation, not a stable size promise.
Logical bytes fall 23.6% for this synthetic fixture, while the operation hot row
is substantially narrower. Distinct payloads do not become duplicates. Archived
body remains in database records, compressed and addressed by verified content
identity; this does not claim the entire records table stops owning tool text.
The extra read is a batch content fetch, not one fetch per returned operation.

Separate manual attachment probe used a 200602-byte model-message payload with
large text and a binary attachment. GC retained its attachment, read-only snapshot
export and restore succeeded, and restored message bytes matched exactly. The
text stayed in the database. The attachment alone used the existing ObjectStore.
CI regression additionally covers rollback, missing content, opaque JSON, small
inline payloads, and cold batch lookup; those tests were added, not run locally.
