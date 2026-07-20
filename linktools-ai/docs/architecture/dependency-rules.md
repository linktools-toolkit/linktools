# Dependency Rules

Source of truth for package dependency direction (plan §3.3, §7.6, AC-02).
Enforced in `tests/ai/architecture/test_dependency_rules.py`; this document is
the human-readable companion.

## Principle

Dependencies point **inward only**. Domain packages define facts and narrow
ports; application packages hold the single orchestration path; infrastructure
implements the ports; the composition root wires them. No domain package may
import a backend or another domain's internals — only stable models or ports.

## Current enforced rule

| Importer | Must not import | Why |
|---|---|---|
| `governance` | `jobs` | Identity/authorization value types must not reach into the job runtime. Holds today; Phase 1 formalizes `identity/`. Package converged from `security` + `policy` in Phase 6 op 1; the job runtime was `task` before Phase 7's rename. |

## Target direction (end state, §3.3)

```
identity        — no deps (PrincipalContext, ActorRef, Scope, ScopeSet)
asset           — no artifact/runtime/providers/registry
artifact        — depends on identity tenant ids + own port; NOT on AssetStore
events          — no _runtime, no concrete Storage backend
jobs            — identity, run stable models, artifact port; NOT _runtime.builder
agent / swarm   — run, tool, model, catalog ports; NOT filesystem/sqlalchemy
governance      — identity, run, tool fact models; NOT agent runner / job worker
_runtime        — may depend on all public ports + concrete impls
```

Other packages must not reverse-import `_runtime.builder`.

## How rules are promoted

Phase 0 records the rules that already hold plus the target table above. As a
phase lands a package boundary, move its `(importer, forbidden)` pair from
`TARGET_RULES_NOT_YET_ENFORCED` into `FORBIDDEN_IMPORTS` in the test file. At
Phase 9 every target rule is enforced and the target table is empty.

## Cycle policy

No two top-level packages may import each other (2-cycle check today; full
cycle detection added as packages stabilize). A cycle is never fixed with a
function-local import — it is fixed by correcting the owner of the concept.
