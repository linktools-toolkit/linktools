# linktools-ai Spec Governance

`AGENTS.md` is the authoritative rule set for this package. Design specs explain a scoped change; they do not override repository or package Required Rules.

## Precedence

1. Repository `AGENTS.md` Required Rules.
2. `linktools-ai/AGENTS.md` Required Rules.
3. Released durable contracts and persisted compatibility boundaries.
4. The latest explicitly approved spec for the affected topic.
5. Older specs, review notes, implementation guides, and archived drafts.

A newer spec supersedes older guidance only inside its declared scope. A filename containing `final`, `verified`, or similar wording does not change this precedence.

## Compatibility rules

- Persisted and replayed data stays explicitly versioned.
- Adding a field is not by itself a version boundary. Additive or defaultable changes may remain in the current version when missing data has one frozen historical meaning.
- Non-semantic metadata must not change semantic digests or idempotency identities.
- Ordinary durable payloads should validate required fields and known values while allowing compatible additive metadata. Reserved envelopes and discriminators may remain strict.
- Truly incompatible semantic changes require a new version and an explicit compatibility or migration path.
- Durable snapshots remain minimal: persist exact dependency semantics only after that semantic fact has actually been established.

## Current Skill/Subagent decisions

The current Skill/Subagent refactor follows these settled rules:

- Skill descriptions are model-visible discovery semantics. When present they belong to the Skill v1 semantic contract and therefore affect capability and Agent-definition identity; older v1 payloads without a description retain their historical identity.
- An Agent's own `description` remains discovery metadata for that Agent declaration and does not change the Agent's execution-definition identity.
- When an Agent is exposed as a Subagent, the parent binding pins the child's model-visible discovery description in its logical `SubagentRef`; changing that description changes the parent binding's discovery semantics without pinning the future child execution definition.
- The existing `AgentBindingSnapshot` v1 wire and digest remain stable for historical payloads that do not contain the additive discovery description.
- `selected_subagents` is a same-version additive field used only when declaration-level selected children differ from execution-level effective delegation targets; older v1 payloads infer it from `subagents`.
- Parent bindings persist logical child Agent references and their established discovery semantics. A child Execution still resolves its Agent definition at delegation time and owns its own exact Agent binding once created.
- One-level child bindings keep declaration-level child selection for exact historical definition recovery but expose no nested delegation targets.
- Vendor-specific retry/tool projection stays in Runtime-private adapters; generic Skill/Subagent capabilities remain vendor-neutral.

## Superseded design patterns

Do not reuse older guidance that requires any of the following when it conflicts with the rules above:

- dropping compatibility merely because a feature has not yet been released;
- bumping a durable version for every additive field;
- exact-key rejection for ordinary extensible payloads;
- recursively pinning future child execution semantics into a parent binding;
- using current mutable defaults or catalogs to redefine an already-established historical durable fact;
- importing SDK-specific retry behavior into vendor-neutral capability code.

Keep this file small. Concrete schemas, owners, state transitions, and implementation details belong in the active scoped spec and code rather than being duplicated here.