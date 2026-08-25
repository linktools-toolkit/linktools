# linktools-ai

`linktools-ai` is the Agent runtime layer for LinkTools. It keeps declaration, discovery, composition, identity, execution, and persistence responsibilities separate:

```text
AssetStore
    -> AssetRepository
    -> AgentSpec
    -> AgentCompiler.compile(...)
    -> AgentDefinition
    -> Runtime.agent(...)
    -> AgentHandle
    -> AgentCompiler.bind(..., output=...)
    -> AgentBinding
    -> Execution / Session turn / TaskGraph / Temporal / Recovery
```

- **Asset** owns declaration origin, storage, discovery, and decoding.
- **AgentSpec** declares one Agent.
- **Capability** supplies executable behavior.
- **AgentDefinition** is the compiled output-independent definition inside one Runtime.
- **AgentBinding** is the exact executable identity for one output contract.
- **Session** is bound to the stable `AgentSpec.id`.
- **Execution**, recovery, evaluation, TaskGraph, and Temporal pin the exact Agent binding.

## 1. Run a workspace

### Command line

```bash
ai-run "review this change" --project /workspace/project --model gpt-4o-mini
python3 -m linktools ai run "review this change" --project /workspace/project --model gpt-4o-mini
```

Useful options:

- `--base-url`, `--api-key`, and `--model` also read `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.
- `--storage filesystem|sqlite` selects Runtime state storage.
- `--planning` enables planning for this execution.
- `--thinking` requests model thinking for this execution when supported.
- `--json` emits one terminal JSON result.

### Python

```python
from linktools.ai.model import ModelRegistry
from linktools.ai.workspace import Workspace, open_workspace_runtime

workspace = Workspace.discover("/workspace/project")
models = ModelRegistry.openai(model="gpt-4o-mini")

async with open_workspace_runtime(workspace, models=models) as runtime:
    agent = runtime.agent()
    result = await agent.run(
        "review this change",
        memory_scope=workspace.workspace_id,
        planning=True,
    )
```

Select a named Agent or pass an inline `AgentSpec`:

```python
from linktools.ai.spec import AgentSpec

named = runtime.agent("audit")
inline = runtime.agent(
    AgentSpec(
        id="reviewer",
        revision=1,
        model="default",
        system_prompt="Review changes carefully.",
        allow_tools=("read_file",),
    )
)
```

`Runtime.agent()` selects Agent identity and optionally adds durable Agent-local Runtime capabilities. Output, planning, and thinking belong to each execution:

```python
from pydantic import BaseModel
from linktools.ai.capability import RuntimeCapability

class ReviewResult(BaseModel):
    summary: str

local_capability = RuntimeCapability.from_spec(
    "review-context",
    MyCapability,
    config={"mode": "strict"},
    revision=1,
)

agent = runtime.agent("audit", capabilities=(local_capability,))
result = await agent.run(
    "inspect the patch",
    output=ReviewResult,
    planning=True,
    thinking=False,
)
```

Agent-local `RuntimeCapability` values must be durable and exactly restorable. Direct Python-only capabilities are suitable for Runtime-global composition because the application recreates them when the Runtime is rebuilt.

## 2. Identity model

A Session is bound to `AgentSpec.id`, not a compiled Agent digest.
Each new execution uses the current Agent binding.
Retry and recovery remain pinned to the exact historical binding.

History is content-owned: a projection digest identifies projected conversation
content, not the Agent definition that consumed it.

## 3. Agent controls

`AgentSpec.allow_tools` is the static upper bound for business/external/model-visible tools. It does not own execution modes or independently disable platform control capabilities such as Skill loading, Subagent delegation, or planning support.

Planning and thinking are execution modes. Planning applies the Runtime's model-visible presentation and admission rules before durable side effects begin.

Subagents are selected from the frozen root Agent catalog. A root Agent may delegate to other root Agents, never itself. Child executions inherit the parent execution's planning/thinking modes and existing memory behavior, do not inherit the parent handle's local Runtime capabilities, use the default assistant-text output contract, and cannot create another subagent layer.

## 4. Asset loading

The default workspace source is the workspace storage root, normally backed by the `.linktools` workspace layout:

```text
agents/<id>
skills/<id>/SKILL.md
mcp/<id>
```

`AssetRepository` is the complete typed Asset composition passed to Runtime. It is constructed directly from an `AssetStore` and `AssetTypeBinding` sequence; callers do not need a public registry or registry snapshot.

For the standard workspace layout, use `build_workspace_assets()`:

```python
from linktools.ai.workspace import build_workspace_assets, open_workspace_runtime

assets = await build_workspace_assets(
    workspace,
    bindings=(my_asset_binding,),
)

async with open_workspace_runtime(
    workspace,
    assets=assets,
    models=models,
    capability_providers=(my_provider,),
) as runtime:
    ...
```

The helper merges built-in bindings with downstream bindings. If a custom binding uses an existing built-in kind, its `value_type` must match that built-in kind. Duplicate or conflicting kinds fail closed.

Applications that need another Asset storage backend can provide an `AssetStore` to the helper:

```python
from linktools.ai.asset import AssetStore
from linktools.ai.workspace import build_workspace_assets

store = AssetStore(my_storage)
assets = await build_workspace_assets(
    workspace,
    store=store,
    bindings=(my_asset_binding,),
)
```

When a custom `store` is supplied, path adaptation belongs to that store/backend composition and `path_adapter` must not also be passed. When the helper creates the default directory store, a custom `AssetPathAdapter` may be supplied there.

A `CapabilityProvider` declares one public `value_type`. At workspace startup it is bound once to all matching discovered Assets. Every Asset binding whose `value_type is AgentSpec` is instead discovered directly into the root Agent catalog, regardless of the Asset kind name, so downstream kinds such as `worker` or `audit` can declare Agents without new Runtime concepts.

## 5. Capability scopes

There are three capability scopes:

1. **Asset-backed Runtime-global capabilities**: `AssetRepository -> CapabilityProvider -> CapabilityBinding`.
2. **Direct Runtime-global capabilities**: supplied through `open_workspace_runtime(..., capabilities=...)` and recreated by Runtime composition.
3. **Agent-local capabilities**: supplied through `runtime.agent(..., capabilities=...)`; they must be durable and exactly restorable.

There is no arbitrary `agent.run(..., capabilities=...)` scope. Runtime-global capability fingerprints and Agent-local capability descriptors participate in the exact executable binding, not Session identity.

## 6. Output

Output is an execution binding, not part of `AgentSpec`, `Runtime.agent()`, or Session identity:

```python
from pydantic import BaseModel

class Finding(BaseModel):
    title: str
    severity: str

agent = runtime.agent("audit")
result = await agent.run("inspect the patch", output=Finding)
```

The binding persists the exact JSON Schema and its fingerprint, not a Python import path. Normal executions use the supplied `BaseModel` and keep Pydantic's validation, coercion, defaults, and validators. Recovery can reconstruct the durable schema contract from the persisted snapshot without registering or importing that Python output type.

## 7. Sessions and history

```python
agent = runtime.agent("audit")
session = await agent.create_session("chat-1")

text_result = await agent.run(
    "hello",
    session_id=session.session_id,
)
structured_result = await agent.run(
    "summarize the same conversation",
    session_id=session.session_id,
    output=Finding,
    planning=True,
)

page = await runtime.session.history(
    session.session_id,
    principal=runtime.default_principal,
    cursor=None,
    limit=100,
)
```

A Session is bound to `AgentSpec.id`, not a compiled Agent digest. Each new execution uses the current Agent binding. Retry and recovery remain pinned to the exact historical binding.

Session history is the committed conversation projection. Execution trace/transcript remain separate audit views.

## 8. Runtime persistence

`RuntimeStatePlan` selects persistence per Runtime domain. The workspace default stores filesystem Runtime state below the workspace storage root.

Filesystem RuntimeState is an embedded single-writer backend. Domains with the same canonical `transaction_root` share one filesystem storage group and can publish a cross-domain checkpoint atomically.

SQL RuntimeState uses optimistic conditional DML. Runtime correctness does not require or use pessimistic row, table, or advisory database locks. SQL routes sharing the same `AsyncEngine` participate in one physical SQL storage group; SQLite routes sharing one canonical database path use the same grouping rule.

For externally managed SQL storage, provision the schema before Runtime startup:

```python
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime import RuntimeState, RuntimeStatePlan, RuntimeStateRoute
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine("sqlite+aiosqlite:////var/lib/my-app/runtime.db")
await provision_runtime_database(engine)

state = RuntimeState.from_plan(
    RuntimeStatePlan(
        conversation=RuntimeStateRoute.sql(engine),
        execution=RuntimeStateRoute.sql(engine),
        recovery=RuntimeStateRoute.sql(engine),
        memory=RuntimeStateRoute.sql(engine),
    )
)
```

MySQL, PostgreSQL, and SQLite use the same Runtime state contracts. The application owns an externally supplied engine and disposes it after Runtime shutdown.

Runtime persistence uses a tolerant reader for ordinary internal dataclasses,
while exact execution and storage contracts remain fail-closed.

## 9. Durable execution binding

Every current V1 execution and recovery input carries a mandatory `AgentBindingSnapshot` with:

- canonical `AgentSpec`
- `agent_digest`
- ordered Agent-local Runtime capability descriptors
- output schema definition, id, revision, and fingerprint
- `binding_digest`

Restore is exact for durable identity: the Runtime restores local capability descriptors, recompiles the Agent, verifies `agent_digest`, reconstructs the persisted output schema contract, rebuilds the binding, verifies `binding_digest`, and requires the reconstructed snapshot to equal the persisted snapshot. Python-only output validation behavior is intentionally process-local and is not persisted as durable identity. There is no root-definition fallback and no partial or null-binding V1 path.

Planning and thinking are mandatory durable execution-mode fields. Retry and fork use the source exact binding and source modes.

## 10. TaskGraph and Temporal

`AgentHandle.task()` writes exactly one Agent TaskGraph V1 input:

```text
type = "linktools.ai.agent"
version = 1
binding = <AgentBindingSnapshot payload>
user_prompt
planning
thinking
```

The binding snapshot is the sole durable Agent-binding authority in the node input; duplicate Agent id/digest/binding-digest fields are not stored alongside it. Unknown, legacy, partial, or corrupt V1 shapes are rejected.

Temporal execution request transport also has one V1 shape:

```text
version = 1
user_prompt
principal
idempotency_key
memory_scope
planning
thinking
binding = <AgentBindingSnapshot payload>
```

Workflow state may keep the compact `binding_digest`; loading the persisted request requires `binding.binding_digest` to match that state. Applications integrating Temporal should use the public `WorkflowGateway` and worker components rather than parsing transport objects directly.

## 11. Public composition boundary

`open_workspace_runtime()` accepts already-composed public dependencies:

```python
async with open_workspace_runtime(
    workspace,
    tenant_id="tenant",
    assets=assets,
    state=state,
    models=models,
    capability_providers=(my_provider,),
    capabilities=(my_runtime_capability,),
) as runtime:
    ...
```

Asset construction knobs do not belong on `open_workspace_runtime()`: construct an `AssetRepository` first, normally through `build_workspace_assets()`.

`Runtime.agent()` accepts only an Agent id/spec plus Agent-local capabilities. Execution output/modes belong to `AgentHandle.start()`, `run()`, `stream()`, `task()`, and evaluation methods as appropriate.

Top-level `linktools.ai` remains intentionally small. Sibling packages use the public `linktools.ai.agent` subsystem boundary rather than importing another package's private modules.

## 12. Identity boundaries

| Field | Meaning |
|---|---|
| Runtime `namespace` | Isolates one Runtime data set inside a storage target |
| `tenant_id` | Authorization and resource ownership boundary |
| `AgentSpec.id` | Stable logical Agent identity for a Session |
| `agent_digest` | Exact output-independent Agent definition |
| `binding_digest` | Exact executable Agent + output identity |
| `memory_scope` | Selects a memory collection inside one tenant |
| Asset `namespace` | Isolates raw Asset storage; unrelated to Runtime state |
| Asset `kind` | Logical typed declaration kind |
| Task/Tool `owner` | Current lease holder |

`open_workspace_runtime()` uses `workspace.workspace_id` as the Runtime namespace. Its optional `tenant_id` defaults to `default`.

## 13. Development checks

From the repository root:

```bash
PYTHONPATH=linktools-ai:linktools-ai/src:linktools/src python3 -m compileall -q linktools-ai/src/linktools/ai
PYTHONPATH=linktools-ai:linktools-ai/src:linktools/src python3 -m pytest -q tests/ai
PYTHONPATH=linktools-ai:linktools-ai/src:linktools/src python3 -m ruff check linktools-ai/scripts/build linktools-ai/src/linktools/ai linktools-ai/src/linktools/commands/ai tests/ai
python3 linktools-ai/scripts/build/architecture.py
```
