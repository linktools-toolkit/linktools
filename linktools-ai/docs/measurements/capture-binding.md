# Frozen capture binding references

Manual SQLite probes, 2026-11-12. Baseline `de3bdc8d`; full capture hooks through
`47e32c2a`, with CI regressions in `9e5ef10c`. These are targeted probes using the
installed project dependencies, not a local execution of the formal test suite.

## Bytes and ownership

The baseline public `capture_input` path stored a complete Agent binding in the
capture record. One baseline fixture had a 57344 B system prompt, a 57912 B binding
JSON value, and a 58434 B complete capture JSON value. The candidate full-path
fixture used a different, 55296 B prompt: its complete stored capture JSON was
1137 B and contained no prompt bytes. These two fixtures demonstrate the storage
shapes, **not a matched percentage reduction or a total-database size comparison**.

Separately, a matched helper-only probe at `47233335` measured a 56510 B value
becoming a 469 B projection. The archive bytes exactly matched the existing
Execution binding codec, and expansion returned the original value. This helper
number is not the size of a complete public capture or the whole database.

A capture owns its fixed binding reference independently of the source Execution.
The body remains in the existing database archive; ordinary text does not enter
ObjectStore. Identical bindings share one body within a physical archive. Split
TASK and EXECUTION storage groups each retain their own physical copy. Reference
acceptance requires exact reconstruction of the original binding JSON; legal
noncanonical values remain inline. Identity, scope, summary, full body bytes and
the original capture digest are checked on recovery.

Actual shared-route and split-TASK-route probes deleted the source Execution,
ran offline GC, read the capture, exported a read-only snapshot, restored it to a
new root, and read the complete binding from a cold store. All passed. An injected
failure after archive publication rolled back both the new capture owner and its
body in the split route; retry succeeded. Cold archive discovery followed by
owner deletion returned INPUT_CAPTURE_UNAVAILABLE without hydrating the stale
body. A separate two-capture probe then removed the source Execution and used
the public offline capture-purge operation on each capture in turn. After the
first purge and GC, the remaining capture still restored its binding and exactly
one binding body remained. After the final purge and GC, no binding body remained
in the TASK archive. Both shared and split SQLite routes passed this check.

## Public read cost

Counts exclude initial setup and capture creation. The public `read_agent` call
includes the expiry-fence read. Both shared SQLite routing and split TASK routing
produced the same counts:

| Read | BEGIN | SELECT | INSERT / UPDATE / DELETE |
| --- | ---: | ---: | ---: |
| Warm archive | 2 | 3 | 0 / 0 / 0 |
| First read after genuine read-only reopen | 4 | 5 | 0 / 0 / 0 |

Warm reads fetch the expiry fence, owner and binding. Cold reads additionally
discover the archive and restart the owner read. The owner and binding bytes are
captured in one read-only database snapshot; no old owner is reused across cold
archive discovery. Attachment restoration follows that database capture. No
ObjectStore or Native-message table reads occurred in these binding-only probes.
These are full current-operation costs, not a claim that all queries are newly
added or that reads became cheaper.
