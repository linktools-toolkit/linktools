# linktools-ai

Domain-agnostic Agent/Session/Registry/Capability runtime, built on
[pydantic-ai](https://ai.pydantic.dev/). Library only — no CLI commands, no
domain-specific business logic; consumers construct agents against this
runtime and supply their own `agent.md`/`SKILL.md`/`mcp.yaml` content and
storage backends.

## Layout

The package is organized by **feature module**, not by technical layer: each
module keeps its Protocol, local implementation, and pydantic-ai capability
together, so a given concern lives in one directory instead of being spread
across three.

```
src/linktools/ai/
  agent.py          agent hierarchy (BaseAgent, LlmAgent, RuntimeAgent, SubAgent)

  core/             non-optional engine skeleton — spec loading (registry.py),
                     prompt assembly (prompt.py), model client construction
                     (model_runtime.py), per-call capability assembly (run.py),
                     execution-context resolution (runtime.py)
  session/          Session types, coordination, context-window/summary
                     policy, and history/artifact storage (local + remote
                     TranscriptStore-backed)
  execution/        file/terminal tool execution — ExecutionBackend Protocol
                     + LocalExecutionBackend, toolset wiring
  skill/            skill definitions (SKILL.md) + the capability that
                     exposes them to the model
  subagent/         subagent definitions (agent.md) + tree-delegation
                     (call_subagent) capability
  mcp/              MCP server definitions + client/toolset wiring
  resource/         ResourceStore: mem -> disk -> remote-backend layered
                     storage for skill/subagent/MCP definition files, used
                     by the registries above
  support/          generic infra shared across modules: config, utils,
                     workspace slot management

  # Opt-in feature capabilities, each gated by a BaseAgent constructor
  # parameter (see "Feature toggles" below):
  security/           destructive shell command blacklist (on by default)
  stuck_loop/         detects a tool call repeating on identical failure,
                       short-circuits further attempts
  periodic_reminder/  injects a one-time reminder once a conversation
                       crosses a context-usage threshold
  budget/             tracks $ cost from token usage, enforces a hard cap
  tool_search/        naive substring search over available tool names
  swarm/              multiple agents sharing a task queue for parallel work
  fork/               branch execution + result collection
```

## Feature toggles

Every opt-in capability is a named keyword argument on `BaseAgent.__init__`
(and forwarded through `SubAgent.__init__`) — there's no separate config
object to look up. Builtin file tools (`list_dir`, `read_file`, `write_file`,
`batch_files`, `apply_patch`) and bash execution are always available; only
opt-in features (security, checkpointing, tracking, etc.) toggle via these
parameters:

```python
RuntimeAgent(
    spec, session, execution_context,     # execution_context built via AgentKernel.build_context(...)
    model_config_resolver=my_model_config_resolver,  # Callable[[str], RuntimeModelConfig]
    enable_security_preset=True,          # default; the only default-on toggle
    enable_stuck_loop_detection=False,
    enable_periodic_reminders=False,
    enable_tool_search=False,
    budget_usd=None,
)
```

There's no bundled environment object — every value an agent needs
(`root` for `Session.create`, the three registries for `AgentKernel`,
`model_config_resolver` for model config, `workdir` for `RuntimeAgent`) is
an explicit constructor param supplied by the caller. `Session` carries no
path or trace concept at all — `FileSession` has its own `root: Path`;
`RemoteSession` has none. Hook events carry whatever a caller puts in the
`context: dict` passed to `AgentKernel.build_context(..., context={...})`
— the framework has no opinion about what keys it contains (`trace_id` is
just a convention, not a reserved field). See `core/runtime.py`'s
`AgentKernel` and `session/types.py`'s `Session.create`/`agent.py`'s
`RuntimeAgent.__init__` for the exact signatures.

Every other toggle defaults to off/`None`/empty and leaves existing behavior
unchanged; enabling one attaches the corresponding capability, nothing else.
A handful of forward-compatibility parameters (`enable_checkpointing`,
`checkpoint_store`, `enable_plan_mode`, `enable_memory`, `fallback_models`,
`context_files`) are accepted and stored but not yet wired to any capability — they
exist so a follow-up plan can add `plan/`/`memory/`/`checkpoint/` without touching
this constructor signature again. `task_queue` is wired (participates in swarm).

Known gap: `budget_usd` is wired but currently inert — `BudgetTracker` needs
`cost_per_1k_input_tokens`/`cost_per_1k_output_tokens` to actually enforce a
cap, and there's no constructor path to supply them yet. `fallback_models` and
`context_files` remain genuinely inert.

## Design docs

- `docs/superpowers/specs/2026-07-01-linktools-ai-extraction-design.md` —
  original extraction into this sub-package.
- `docs/superpowers/specs/2026-07-02-linktools-ai-harness-modularization-design.md` —
  the feature-module layout and harness-parity capability set this README
  describes.
- `docs/superpowers/plans/2026-07-02-linktools-ai-restructure.md` and
  `docs/superpowers/plans/2026-07-02-linktools-ai-hook-features.md` — the
  implementation plans that executed the design above.

## Tests

Tests live at the monorepo root under `tests/ai/` (not inside this sub-package),
alongside `tests/` for `linktools` core.

```bash
# from the repo root
PYTHONPATH="linktools-ai/src:linktools/src" python -m pytest tests/ai/ -q
```

## Production readiness (Runtime / vNext runtime surface)

The sections above describe the older `LlmAgent`/`RuntimeAgent`/`AgentKernel`
capability layer. The matrix below covers the separate, currently-active
`Runtime`/`AgentRunner`/`SwarmRunner`/`storage.facade` surface
(`linktools.ai.runtime.Runtime.build()` onward) that the `tests/ai/` suite
actually exercises today.

| Capability | Status | Notes |
|---|---|---|
| Single Agent run | Verifiable | Main path closed: CAS state transitions, real cancellation (RunController), Approval pause atomicity. |
| Agent concurrent runs | Verifiable | Backed by DB-level CAS on `RunStore.transition`; relies on test coverage, not a formal load-test. |
| Agent approval pause/resume | Verifiable | Approval persists atomically with checkpoint/transition/events on SqlAlchemy storage; DB-unique-constraint-backed dedup on `(run_id, tool_call_id)`. |
| Swarm basic execution | Verifiable | `SwarmRunner` reuses the SAME `AgentRunner` `Runtime.build()` assembles for top-level runs -- worker Runs get identical Tool/Policy/Middleware/UoW wiring. |
| Swarm cancel | Verifiable | Propagates through `RunController` to the driving run and every active child run (real `task.cancel()`, not just a store-status flip) when a controller is wired -- falls back to store-only cancel otherwise. |
| Swarm task completion (`complete_task`/`fail_task`) | Verifiable | `expected_version` is mandatory; conditioned on `status='claimed'` and (optionally) `active_run_id`, so a reclaimed-and-superseded worker cannot clobber a new owner's result. |
| Event run stream | Verifiable | `stream_id == run_id` for every current caller. |
| Event custom (non-run) stream | Not yet supported | `EventEnvelope.stream_id` is a first-class field with its own `(stream_id, sequence)` DB constraint, but nothing currently mints a `stream_id != run_id`. |
| `FileStorage` | Dev / single-process | No cross-process consistency guarantee; documented throughout the store implementations. |
| `SqlAlchemyStorage` | Verifiable | CAS/UoW paths are unit-tested against SQLite; no Postgres integration test exists in this repo. |
| `SqlAlchemyStorage` UoW + concurrent same-session writers | Not supported | A single UoW-mode `SessionStore.append_messages` writer is fine; two concurrent UoW-mode writers to the SAME session in the SAME transaction are not (see `storage/sqlalchemy/session.py` docstring) -- would need a dedicated sequence-counter table. |
| Resource multi-backend `propfind` pagination | Verifiable | Regression-tested for shadow/whiteout/multi-overlay/small-`limit` pagination; still the simple literal-path cursor, not the spec's opaque token. |
| `WorkspaceManager` | Experimental | `WorkspaceManager`/`LocalWorkspaceManager` exist but have no consumer in `Runtime`/`AgentRunner` -- not wired into the main path until a real caller needs it. |

Known limitations worth calling out explicitly:

- `FileStorage` is single-process only; do not point two processes at the same root.
- Swarm cancel only stops what it can see: cross-process children (a different process's `RunController`) still only get the store-level `CANCELLING`/`CANCELLED` fallback.
- Event streams are 1:1 with runs today regardless of the `stream_id` field's existence.
- `SqlAlchemyStorage` does not use SAVEPOINT-based (`session.begin_nested()`) conflict isolation anywhere (`SqlAlchemyIdempotencyStore.reserve`, `ApprovalStore.create_or_get_pending`) due to a known aiosqlite limitation: a savepoint that releases cleanly does not reliably participate in a *later*, unrelated failure's rollback of the enclosing transaction (see either method's docstring for the full writeup). A genuine unique-key collision inside an explicit UnitOfWork therefore aborts the whole transaction rather than gracefully continuing within it -- acceptable given the current codebase's call sites never race the same key within the same transaction. Worth re-testing against a real Postgres/asyncpg backend if `begin_nested()` isolation is reconsidered there.

## Capability Runtime, Provider boundary & optional deps

`linktools-ai` is a **Capability Runtime**: an `AgentSpec`'s declared `tools`
are resolved into prompt sections + toolsets via pluggable capability providers.
The Runtime depends only on **Provider Protocols**, never on a concrete
Registry — a Registry is merely the default file/resource-backed Provider
implementation. Downstream systems can bypass the default formats entirely and
supply any backend (DB, config center, HTTP API, git, object storage) that
returns the standard Specs.

```
AgentSpec.tools -> CapabilityAssembler -> CapabilityProvider(s)
  builtin / skill / mcp / subagent / package(-resource/-entrypoint)
  -> CapabilityBundle (prompt_sections + toolsets)
  -> AgentRunner / ToolExecutor / Middleware / EventStore
```

- **Providers** (`linktools.ai.providers`): `AgentSpecProvider`,
  `SkillSpecProvider`, `MCPServerSpecProvider`, `ToolPolicyProvider`,
  `SwarmSpecProvider`, `SubagentSpecProvider`, `PackageSpecProvider`,
  `PackageResourceProvider` — all `async list_ids()/get(...)`. Pass them to
  `Runtime.build(...)` either as a `ProviderBundle` or as the expanded
  `agents=`/`skills=`/`mcp_servers=`/... params (not both).
- **Default registries** (`linktools.ai.registry`) implement these Protocols and
  parse the recommended formats (`agent.md`, `SKILL.md`, `mcp.yaml`,
  `tool.yaml`, `swarm.yaml`). Format-clarifying aliases (`MarkdownAgentRegistry`,
  `YamlMCPRegistry`, ...) make clear these are *recommended*, not mandatory.
- **Tool-call exposure is controlled** (`CapabilityToolExposurePolicy`): default
  minimal — prompt catalog + read-only discovery on; execution tools off until
  opted in; per-capability and total tool caps; `mcp:*` wildcard gated by
  `allow_mcp_wildcard`; package entrypoints list-only unless `expose_call_tool`.
- **Package-scoped capability** (`linktools.ai.package`): `skill-creator`-style
  bundles (resources + entrypoints) resolve under a `PackageScope` so the same
  entrypoint name in two packages never collides (`package:<id>:agent:<name>`).
  Resources are read through a path-sandboxed, paginated, size-clamped provider.
- **base_url is literal**: model `base_url` is passed through verbatim — custom
  gateway paths are never corrupted. Opt into the legacy auto-`/v1` behavior with
  `RuntimeModelConfig.base_url_mode="append_v1_if_missing"`.
- **Runtime is an async context manager** (`async with Runtime.build(...) as rt`)
  that releases MCP connections on exit.

### Optional dependencies (storage backends)

Core install pulls in only pydantic-ai + the linktools core — **no SQLAlchemy**.
`FileStorage` and the Store/Provider Protocols work out of the box:

```bash
pip install linktools-ai
```

`SqlAlchemyStorage` requires an optional extra:

```bash
pip install "linktools-ai[sqlite]"      # SQLAlchemy + aiosqlite (default DB)
pip install "linktools-ai[postgres]"    # SQLAlchemy + asyncpg
pip install "linktools-ai[mysql]"       # SQLAlchemy + asyncmy
pip install "linktools-ai[all]"         # every backend
```

`import linktools.ai` and `import linktools.ai.storage` succeed without any
extra; accessing `SqlAlchemyStorage` then raises an `ImportError` with the
install hint. Storage can be reused as-is, composed store-by-store, or fully
reimplemented behind the Store Protocols + contract tests
(`tests/ai/contracts/`, `tests/ai/storage/contract/`).

