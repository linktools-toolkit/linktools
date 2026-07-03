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
  resource_store/   ResourceStore: mem -> disk -> remote-backend layered
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
