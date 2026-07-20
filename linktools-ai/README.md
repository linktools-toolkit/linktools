# linktools-ai

Agent/Session/Capability runtime built on
[pydantic-ai](https://ai.pydantic.dev/). Library only — no CLI commands, no
domain-specific business logic. Consumers declare agents, skills, and MCP
servers via specs, wire a Storage backend, and call `Runtime.run`.

## Quick start

```python
from linktools.ai import Runtime
from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.storage.facade import FileStorage

async with Runtime.build(
    storage=FileStorage(root="./data"),
    model_router=my_model_router,
) as rt:
    spec = AgentSpec(
        id="writer",
        name="writer",
        model=ModelPolicy(primary="gpt-4o"),
        instructions=PromptSpec(instructions="You are a careful writer."),
        tools=(ToolRef(kind="builtin", name="file-read"),),
    )
    result = await rt.run(spec, "Write a one-line summary of the project.")
    print(result.output)
```

## Architecture

```
Runtime.build(storage, model_router, providers, options, ...)
  -> AgentCompiler (compiles AgentSpec into a pydantic-ai Agent)
  -> AgentEngine (drives the agent: model calls, tool calls, governance)
  -> SwarmRunner (orchestrates multi-agent swarms)
  -> ManagedToolAdapter (single governance entry: policy + security pipeline)
  -> ToolExecutor (single execution entry: policy-check then handler)
  -> Storage (File or SQLAlchemy backends for runs/sessions/events/...)
```

### Single execution path

Every tool call goes through exactly one chain:

```
Model -> ManagedToolAdapter -> ToolExecutor.check (policy/approval)
                               -> SecurityPipeline.before_tool
                               -> handler (timeout + retry)
                               -> SecurityPipeline.after_tool
```

There is no raw-tool bypass and no non-managed path.

### Approval pause/resume

When policy says `REQUIRE_APPROVAL`, the executor raises `RunPaused`. The
runner persists the `ApprovalRequest` + checkpoint + transitions the Run to
`WAITING_APPROVAL` atomically (on SqlAlchemy storage, one transaction). On
`File` storage, the approval write is mandatory — a failure ends the Run
`FAILED` rather than leaving it half-paused.

`Runtime.resume(run_id)` restores the **original** spec + identity from a
persisted `RunDefinitionSnapshot` — a caller cannot inject a different model,
tool list, or identity on resume.

### MCP

MCP servers are declared via `mcp.yaml` specs. Strict discovery (the default)
fails closed when tools cannot be enumerated. `enabled_tools` / `disabled_tools`
/ `tool_prefix` shape the exposed toolset. Connection caching keys on a
canonical fingerprint of the server config (transport, command/url, env digest,
header digest, filters) so a rotated secret invalidates the cache without the
plaintext ever entering the key.

### Swarm

`SwarmSpec` declares agents + a coordinator + a strategy + limits. The
`SwarmRunner` compiles member agents, distributes tasks, and aggregates results.
Cost and token limits (`max_total_cost`, `max_total_tokens`) can cap execution.

## Provider + Registry

`Runtime.build` accepts a `ProviderBundle` of spec providers (agents, skills,
MCP, tools, swarms, subagents, packages). Default registries parse
`agent.md` / `SKILL.md` / `mcp.yaml` / `tool.yaml` / `swarm.yaml` from a
filesystem root or a `ResourceStore`.

Custom providers implement the same Protocols — the Runtime depends on
Protocols, never on a concrete Registry.

## Storage

| Backend | Install | Cross-store transactions |
|---|---|---|
| `FileStorage` | `pip install linktools-ai` | No (sequential writes) |
| `SqlAlchemyStorage` | `pip install "linktools-ai[sqlite]"` | Yes (UoW / single transaction) |

`import linktools.ai` succeeds without SQLAlchemy. Accessing
`SqlAlchemyStorage` raises an `ImportError` with an install hint.

## Domain invariants

Domain models (`AgentSpec`, `ToolDescriptor`, `ModelPolicy`, `MCPServerSpec`,
`CapabilityRef`, etc.) validate their contract at construction — a custom
provider that builds one directly cannot create an invalid object. Mapping
fields are deep-frozen.

Registry parsing uses a `StrictConfigReader` that distinguishes a missing field
(uses default) from an explicit `null` (rejected) and rejects unknown fields,
so a typo like `agentd_id` cannot be silently ignored.

Canonical JSON (`linktools.ai.json.canonical_json`) is used for every hash and
fingerprint path — `default=str` is forbidden (it silently coerces arbitrary
objects into unstable/colliding strings).

## Tests

```bash
# from the repo root
PYTHONPATH="linktools-ai/src:linktools/src" python -m pytest tests/ai/ -q
```
