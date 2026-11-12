# Fixed configuration and empty diagnostics

## Matched Runtime measurement

The existing two-turn Runtime/FunctionModel workload was run without changing its
script, storage helper, inputs, Python, or dependencies. All five recorded hashes
and environment descriptions match. This measures all business table columns,
not just `records`; it is not the formal test suite.

Small workload, in order: total business bytes; runtime / warm / cold / reopen
SELECT + WITH counts:

| Revision | Business bytes | Runtime | Warm | Cold | Reopen |
| --- | ---: | ---: | ---: | ---: | ---: |
| `6e907b06` baseline | 142632 | 324 | 45 | 48 | 3 |
| `3596f282` fixed-owner first version | 154925 | 415 | 65 | 72 | 4 |
| `00fe7f94` empty diagnostics and observed policy reuse | 153290 | 403 | 63 | 70 | 4 |
| `0229fd50` narrow producer authority | 153290 | 387 | 63 | 70 | 3 |
| `02d8b0db` batched configuration capture | 153290 | 373 | 61 | 68 | 3 |

The large workload at `02d8b0db` measured 153126 business bytes and the same
373 / 61 / 68 / 3 reads. Its baseline was 144571 bytes and 326 / 49 / 56 / 3.
These improvements do **not** meet the overall baseline: the small workload still
uses 10658 more business bytes and 49 more runtime reads. Remaining fixed-owner
metadata overhead must not be described as a storage reduction merely because
`records` is smaller.

## What changed

- Explicit typed `terminal_details: null` means exactly empty safe details and no
  diagnostics. It creates no payload owner or large scoped-reference wrapper.
  Nonempty execution and handoff diagnostics keep the fixed content owner.
- Observed CAS boundaries may retain an unchanged effective policy's original
  reference and overrides. They require the matching observed projection and
  still CAS the physical owner revision. Ordinary reads and policy changes do
  not use this shortcut.
- Dispatch authority does not consume `durable_trace`. Its control projection
  no longer loads that policy field. Terminal history/seal checks still do.
- Binding and policy metadata are captured together in the actual owner scope.
  The content loader is bound to the captured identities, not arbitrary supplied
  records. Cached decoded bytes never replace current source metadata checks.

## Independent checks and remaining budget

Targeted diagnostic scripts verified empty/safe/diagnostic execution and handoff
full reads, metadata, timeline, retention, GC, and readonly snapshot restoration.
The observed boundary emitted no SELECT and rejected a stale cancellation.
All twelve dispatch/settlement cases used one SELECT. A reopened full mutation
kept lazy archive discovery and invoked its callback once.

After configuration batching, binding and policy deletion or body corruption
following an earlier read in the same writable scope was rejected in all four
cases. The bounded model-update flow improved to six SELECT and two BEGIN, but
its existing five-SELECT budget is still unmet. The six reads are authorization
header, current owner, configuration owners, run admission, counters, and model
metadata. No safety read was removed and that budget assertion was not raised.
Formal regressions were added or updated for natural CI; no local suite was run.
