# Glossary

Terms the refactor uses with a fixed meaning (plan §4.3, §4.4). Ambiguity in
these names is what the refactor removes.

| Term | Meaning | Retired alias |
|---|---|---|
| Asset | Pre-run, configuration or content source read via `AssetStore` (path-addressed, overlay/whiteout/CAS). Replaces the public use of `Resource`. | `Resource`, `ResourceStore` |
| Artifact | An immutable, content-addressed (SHA-256) output of a run, recorded with tenant/producer/provenance. Distinct model, store and protocol from Asset. | (was built on `ResourceStore`) |
| Catalog | A stable, strictly-parsed collection of Specs: `list/get/revision`. Static configuration, not runtime wiring. | `Registry` (for static specs), `AgentRegistry` |
| Registry | Runtime implementation registry only (e.g. capability provider kind → factory). | top-level `registry/` (mixed role) |
| Source | Raw bytes/text backing a Catalog. | — |
| Codec | Strict parse + validate for a Catalog item. | — |
| Resolver | Resolves declarations + implementations into executable dependencies. | — |
| Identity | Principal/Actor/Scope value types. Owns `PrincipalContext`, `ActorRef`, `Scope`, `ScopeSet`. | (was in `task.models` / `security.principal`) |
| Governance | Authorization + guardrail + audit decisions (`AuthorizationService`, `GuardrailPipeline`). Decision-only; approval lifecycle stays in `run`. | `security`/`policy` decision code |
| Storage facade | The single aggregate entry point exposing `assets`, `artifacts`, stores, `transactions`, `coordination`, `features`. Aggregates Protocols only — does not construct backends. | — |
| StorageFeatures | Transaction/coordination scope + streaming/leasing/fencing capability flags. Drives capability gating, never `isinstance`. | `StorageCapabilities` |
| Extension | Runtime extension unit (declaration + entrypoint + capability). | `package/`, `Package*` |
| Jobs | The Job/Task/Attempt runtime: lease, heartbeat, fencing, journal, recovery, cancellation. | `task/`, `TaskRuntime/Worker/Store` |
| Execution | Workspace + local/container/sandbox execution backends. The package is **not** renamed; only the isolation backend sits under `execution/sandbox/`. | (prior plan's "Execution → Sandbox" is reverted) |
| RunDispatcher | Narrow protocol jobs/swarm/subagent use to launch sub-runs; breaks the lazy-runtime cycle. | lazy runner closure |
