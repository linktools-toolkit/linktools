# AGENTS.md (tests/ai)

Test rules for `linktools-ai`. Repository and package rules also apply.

## Required Rules

- Test observable behavior or a documented long-lived invariant. Do not freeze private signatures, file layout, helper names, historical API absence, or exact internal call counts unless they are themselves an accepted contract.
- White-box tests are allowed only when the invariant cannot be observed through a stable boundary; assert the invariant or failure semantics, not the incidental mechanism used to implement it.
- Cover one representative case per equivalent failure class. Add more cases only when they exercise distinct semantics or a proven regression boundary.
- Durable compatibility tests cover committed wire semantics: additive ordinary fields remain readable where the decoder is forward-compatible; unknown versions/types and malformed known fields fail closed.
- Backend parity belongs in shared/parameterized contract coverage. Backend-specific tests cover only behavior that is genuinely backend- or dialect-specific.
- Name tests by the behavior they protect, not by review rounds, fixes, closure phases, or implementation history.
