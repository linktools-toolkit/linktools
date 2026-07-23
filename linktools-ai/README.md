# linktools-ai

Agent / session / capability runtime built on
[pydantic-ai](https://ai.pydantic.dev/). Library only — no CLI commands, no
domain-specific business logic. Consumers declare agents, skills, and MCP
servers via specs, wire a `Storage` backend + a `ModelResolver`, and call
`Runtime.run`.

## Quick start

A tested minimal example lives at [`examples/minimal_runtime.py`](../examples/minimal_runtime.py)
(run by `tests/ai/docs/test_readme_examples.py`). It registers a model, builds a
`Runtime` over a `FilesystemStorage`, runs one no-tool agent, and closes it:

```python
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model import ModelPolicy, ModelRegistry, ModelResolver
from linktools.ai.runtime import Runtime
from linktools.ai.storage import FilesystemStorage
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

registry = ModelRegistry()
registry.register("standard", model=my_model)

storage = FilesystemStorage(root="./data")
async with Runtime.build(
    storage=storage,
    model_resolver=ModelResolver(registry=registry),
    commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
) as runtime:
    spec = AgentSpec(
        id="writer",
        name="writer",
        model=ModelPolicy(primary="standard"),
        instructions=PromptSpec(instructions="You are a careful writer."),
    )
    result = await runtime.run(spec, "Write a one-line summary.")
    print(result.output)
```

`build_runtime(...)` is the function form of `Runtime.build(...)` — same
signature, useful when you want a value rather than the class method.

## Public entry points

The surface the README documents and the tests pin
(`tests/ai/architecture/test_public_api_examples.py`):

```text
FilesystemStorage   Storage            AssetStore         ArtifactStore
AgentCatalog        SkillCatalog       CapabilityResolver ModelResolver
Sandbox             GovernedToolInvoker RunCoordinator    Runtime  build_runtime
```

## Architecture

```text
Runtime.build(storage, model_resolver, providers, ...)
  -> AgentCompiler   (resolves ModelPolicy -> ResolvedModel, compiles AgentSpec)
  -> AgentEngine     (drives the agent: model calls, tool calls, governance)
  -> SwarmRunner     (orchestrates multi-agent swarms)
  -> CapabilityResolver  (resolves declared tools into a governed toolset)
  -> GovernedToolInvoker (single execution entry: policy + approval + security)
  -> Storage         (Filesystem or SQLAlchemy backends)
```

### Single execution path

Every tool call goes through exactly one chain:

```text
Model -> GovernedToolInvoker.check (policy / approval)
                              -> security pipeline before_tool
                              -> handler (timeout + retry)
                              -> security pipeline after_tool
```

There is no raw-tool bypass and no non-managed path.

## Sandbox + tools

A `Sandbox` supplies a capability (file / terminal access); the
`CapabilityResolver` exposes it as a tool; every call flows through the
`GovernedToolInvoker`. A run never reaches the sandbox backend directly — only
through a governed builtin tool resolved from the sandbox. See
[`examples/sandbox_runtime.py`](../examples/sandbox_runtime.py):

```python
from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.sandbox.local import LocalSandbox

async with Runtime.build(
    storage=storage,
    model_resolver=ModelResolver(registry=registry),
    sandbox=LocalSandbox(runtime_dir="./workdir"),
    commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
) as runtime:
    spec = AgentSpec(
        id="reader",
        name="reader",
        model=ModelPolicy(primary="standard"),
        instructions=PromptSpec(instructions="You read files the user names."),
        tools=(ToolRef(kind="builtin", name="file-read"),),
    )
    result = await runtime.run(spec, "Read README.md and summarize.")
```

### Approval pause / resume

When policy says `REQUIRE_APPROVAL`, the invoker signals a pause and the runner
persists the `ApprovalRequest` + checkpoint and transitions the Run to
`WAITING_APPROVAL` atomically (on SQLAlchemy storage, one transaction). On
Filesystem storage the approval write is mandatory — a failure ends the Run
`FAILED` rather than leaving it half-paused. The run's paused outcome is the
explicit `RunStatus.PAUSED` result.

`Runtime.resume(run_id)` restores the **original** spec + identity from a
persisted `RunDefinitionSnapshot` — a caller cannot inject a different model,
tool list, or identity on resume.

### MCP

MCP servers are declared via `mcp.yaml` specs. Strict discovery (the default)
fails closed when tools cannot be enumerated. `enabled_tools` /
`disabled_tools` / `tool_prefix` shape the exposed toolset. Connection caching
keys on a canonical fingerprint of the server config so a rotated secret
invalidates the cache without the plaintext ever entering the key.

### Swarm

`SwarmSpec` declares agents + a coordinator + a strategy + limits. The
`SwarmRunner` compiles member agents, distributes tasks, and aggregates
results. Cost and token limits (`max_total_cost`, `max_total_tokens`) can cap
execution.

## Provider + Catalog

`Runtime.build` accepts a `RuntimeDependencies` bundle of spec providers
(agents, skills, MCP, subagents, packages). Default catalogs parse `agent.md` /
`SKILL.md` / `mcp.yaml` from a filesystem root or an `AssetStore`:

- `AgentCatalog` / `SkillCatalog` / `MCPCatalog` — the parsed spec catalogs.
- Custom providers implement the same Protocols; the Runtime depends on
  Protocols, never on a concrete catalog.

## Storage

| Backend | Install | Cross-store transactions |
|---|---|---|
| `FilesystemStorage` | `pip install linktools-ai` | No (sequential journaled writes) |
| `SqlAlchemyStorage` | `pip install "linktools-ai[sqlite]"` | Yes (UoW / single transaction) |

`import linktools.ai` succeeds without SQLAlchemy. Accessing
`SqlAlchemyStorage` raises an `ImportError` with an install hint.

## Domain invariants

Domain models (`AgentSpec`, `ToolDescriptor`, `ModelPolicy`, `MCPServerSpec`,
`CapabilityRef`, ...) validate their contract at construction — a custom
provider that builds one directly cannot create an invalid object. Mapping
fields are deep-frozen.

Registry parsing uses a `StrictConfigReader` that distinguishes a missing field
(uses default) from an explicit `null` (rejected) and rejects unknown fields,
so a typo like `agentd_id` cannot be silently ignored.

Canonical JSON (`linktools.ai.json.canonical_json`) is used for every hash and
fingerprint path — `default=str` is forbidden (it silently coerces arbitrary
objects into unstable / colliding strings).

## Tests

```bash
# from the repo root
PYTHONPATH="linktools-ai/src:linktools/src" python -m pytest tests/ai/ -q
```
