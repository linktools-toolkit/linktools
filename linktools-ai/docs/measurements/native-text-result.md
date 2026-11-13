# Native terminal text references

Manual SQLite measurements on the implementation through 9520a1cb, 2026-11-12.
Reproduce with installed project dependencies, without invoking pytest:

```sh
python tests/ai/_measure_native_text_result.py /path/to/checkout
```

The fixture uses an 84002-byte canonical JSON string. Real `StateCommands`
terminal publication writes a fixed Native message reference in the execution
result. The result is rebuilt as the original JSON StoredPayload, checking its
encoding, string type, digest and byte length. Only equal large plain-text
results qualify; the existing 64 KiB policy is unchanged. Small, transformed,
structured and unsupported results retain their existing storage path.

Observed SQL, excluding initial Persistence setup and execution creation:

| Phase | BEGIN | SELECT | INSERT | UPDATE | DELETE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Terminal publication | 3 | 15 | 20 | 8 | 4 |
| First result read after genuine reopen | 3 | 4 | 0 | 0 | 0 |
| Subsequent fresh result read | 1 | 2 | 0 | 0 | 0 |

Cold reads include archive discovery and the read-only snapshot restart; the
probe does not warm the archive first. The two steady-state reads are the owner
and fixed Native range. Terminal publication verifies actual source bytes with
a range read before accepting the reference. The publication row is the whole
operation, not an assertion that all its SQL is newly added or reduced. Native
owner-edge DML and physical-root resolution remain visible in that total.

The probe also passes an existing root containing different text while claiming
the target output's digest and size. The writer rejects it with
STORAGE_INTEGRITY_ERROR, leaving the target execution STARTED. Cold full-record
and timeline result reads round-trip exactly.

Separate minimal probes confirmed that a failure after owner CAS and Native body
publication rolls back both, and that a 3-message interior slice reserves only
its selected 20765-byte message before decompression. A rejecting budget callback
fails before hydration. Structural ancestor rows are not mistaken for bodies.
Snapshot preflight checks temporary Native decoding against the available limit;
final inventory accounting retains each physical body once rather than charging
this preliminary decode as another persisted copy.

CI regressions cover these boundaries in test_native_text_result_runtime.py;
formal test suites were not run locally. Snapshot/restore and final-owner GC
closure have separate regression coverage. This step does not claim all tool
results or structured terminal outputs have been deduplicated with Step content.
