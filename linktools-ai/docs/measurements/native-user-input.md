# Native plain-text accepted input

The narrow optimization applies only to `StoredUserInput(codec="text")` with no
view, an inline UTF-8 payload larger than the existing 64 KiB policy, and a newly
published `UserPromptPart` whose string exactly reconstructs the full accepted
input. Admission still saves the input first. Only real same-group publication
can replace its body reference atomically, using physically read fixed Native
source bytes. Small, structured, attachment-bearing, task, different-content,
and view-bearing inputs remain unchanged. Unknown additional original encoded
metadata also retains the original archive representation.

The optimization does not imply coverage of all 25 inputs totaling 233,637 bytes
in the supplied production distribution. Their average is about 9.3 KiB; the
count actually exceeding 64 KiB was not provided.

Reproduce the standalone SQLite measurement with project dependencies:

```
PYTHONPATH=.:linktools-ai/src:linktools/src python -m tests.ai._measure_native_input
```

On 2026-11-12, a real StateCommands terminal publication with 140,000-byte input
and 140,002-byte JSON text output produced both Native references. Exact input
and output recovery succeeded. Query counts were:

| Read | SELECT | WITH range query | BEGIN |
| --- | ---: | ---: | ---: |
| Warm full `get_fresh`, input and output | 3 | 1 | 2 |
| Reopened cold full `get_fresh` | 5 | 1 | 4 |
| Metadata only | 1 | 0 | 1 |
| Native output result only | 1 | 1 | 1 |

Each WITH is one real SQL query. Full Native input and output hydrate in one
range batch, not separate source queries. Full inspection retains the existing
observation/cache contract and then rereads owner plus fixed content coherently.
The cold path includes archive discovery/retry. A separate small-output probe
used one SELECT and one BEGIN for result-only reads despite its large input.

The existing timeline contract contains the complete accepted input, so timeline
reads still restore it. Metadata/control and result-only reads do not hydrate
input bodies. An input-only full read can add a Native range query relative to
co-located binding/input ContentArchive reads; this is an explicit storage/IO
tradeoff, not a claim of universally lower query counts.

Independent probes verified direct and prepared non-terminal projection followed
by terminal CAS without losing the input reference. An injected failure after
the actual Native-input owner replacement rolled back publication and retained
the original accepted input and STARTED execution. Source roots remain immutable
and are retained through existing Native owner edges, not current checkpoint
pointers. Old archive content awaits existing shared-content GC; no immediate
SQLite file-size reduction is claimed. No ordinary text is sent to ObjectStore.

CI cases are in `tests/ai/test_native_input_runtime.py` and the snapshot closure
regressions. Formal suites were not run locally.

## Snapshot and final-owner collection

An independent real SQLite probe with the 140,000-byte input verified:

- Before GC, the old execution-payload archive remained one row with 500 bytes
  of compressed body, excluding metadata. Existing GC removed that row; the
  logical original input size remains 140,000 bytes, not 500 bytes.
- Read-only export before that GC produced a 16,032-byte snapshot manifest and
  did not include the unreferenced old archive body. Restore recovered the exact
  original StoredUserInput.
- With other Native owners removed, the input's execution owner alone retained
  one message. Deleting the final owner and collecting left zero Native messages
  and zero Native sequences.

These observations distinguish logical dependency removal, archive row
collection and physical SQLite file size; no file-shrink claim is made.
