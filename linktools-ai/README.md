# linktools-ai

`linktools-ai` is the Agent runtime layer for LinkTools. It keeps declaration, discovery, composition, execution, and persistence responsibilities separate:

```text
AssetStore -> AssetRepository -> AgentCompiler -> AgentDefinition -> Runtime
```

- **Asset** stores and resolves typed declarations such as Agent, Skill, MCP, and downstream-defined types.
- **Capability** turns runtime-global behavior or matching Assets into executable capability bindings.
- **AgentCompiler** freezes one effective Agent definition.
- **Runtime** owns sessions, executions, recovery, TaskGraph, and the public Agent handle.

Agent definitions are immutable after compilation. Durable executions persist the local definition data required to restore an inline or overridden Agent after restart instead of falling back to a different current definition.

## 1. Run a workspace

### Command line

```bash
ai-run "review this change" --project /workspace/project --model gpt-4o-mini
python3 -m linktools ai run "review this change" --project /workspace/project --model gpt-4o-mini
```

Useful options:

- `--base-url`, `--api-key`, and `--model` also read `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.
- `--storage filesystem|sqlite` selects Runtime state storage.
- `--planning` enables planning mode.
- `--thinking` requests model thinking when the selected model supports it.
- `--json` emits one terminal JSON result.

### Python

```python
from linktools.ai.model import ModelRegistry
from linktools.ai.workspace import Workspace, open_workspace_runtime

workspace = Workspace.discover("/workspace/project")
models = ModelRegistry.openai(model="gpt-4o-mini")

async with open_workspace_runtime(workspace, models=models) as runtime:
    result = await runtime.agent().run(
        "review this change",
        memory_scope=workspace.workspace_id,
    )
```

Select a named Agent or pass an inline `AgentSpec`:

```python
from linktools.ai.spec import AgentSpec

named = runtime.agent("audit")
inline = runtime.agent(
    AgentSpec(
        "reviewer",
        1,
        "default",
        system_prompt="Review changes carefully.",
        allow_tools=("read_file",),
    )
)
```

Agent-local output and durable capabilities are explicit handle overrides:

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
agent = runtime.agent(
    "audit",
    capabilities=(local_capability,),
    output=ReviewResult,
    planning=True,
    thinking=False,
)
```

A local `RuntimeCapability` must be durable (`RuntimeCapability.from_spec(...)`). Direct Python-only `RuntimeCapability` instances are reserved for runtime-global composition because they cannot be reconstructed from persisted execution state.

## 2. Agent controls

`AgentSpec.allow_tools` has one responsibility: it is the static upper bound for business/external/model-visible tools.

It does **not** activate or disable control capabilities such as:

- Skill `list_skills` / `load_skill`
- Subagent `delegate_task`
- Planning `write_plan`

Planning and thinking are execution modes, not tool allowlists. Planning applies both a model-visible presentation filter and a hard admission gate before durable tool side effects begin.

Subagents are selected from the frozen root Agent catalog. A root Agent can delegate to other root Agents, never itself. Child executions inherit `planning` and `thinking`, do not inherit parent-local Runtime capabilities, and cannot create another subagent layer.

## 3. Asset loading

The default workspace source is `<project>/.linktools`:

```text
.linktools/
├── agents/<id>
├── skills/<id>/SKILL.md
└── mcp/<id>
```

`open_workspace_runtime()` discovers every registered Asset kind. Built-in Agent, Skill, and MCP bindings are always present; downstream applications can add more logical Asset types:

```python
async with open_workspace_runtime(
    workspace,
    models=models,
    asset_bindings=(my_asset_binding,),
    capability_providers=(my_provider,),
) as runtime:
    ...
```

A `CapabilityProvider` declares one public `value_type`. At workspace startup it is bound once to all matching discovered Assets. A provider with no matching Assets is inactive and does not affect Runtime execution identity.

Agent Asset types are deliberately excluded from the provider route. Every Asset binding whose `value_type` is `AgentSpec` is discovered into the root Agent catalog directly.

### Custom Asset storage

Applications that need a database, remote service, or another loading policy can provide an initialized public `AssetStore`:

```python
from linktools.ai.asset import AssetRepository, AssetStore, AssetTypeRegistry
from linktools.ai.storage import StorageOverlay

backend = MyAssetBackend(...)
store = AssetStore(StorageOverlay(backend, writer=backend))
await store.initialize()

registry = AssetTypeRegistry()
registry.register(my_binding)
assets = AssetRepository(store, registry.freeze())
```

Do not import another package's private module or call private backend methods.

## 4. Runtime-global capabilities

Runtime-global behavior is supplied at workspace composition:

```python
async with open_workspace_runtime(
    workspace,
    models=models,
    capabilities=(my_runtime_capability,),
) as runtime:
    ...
```

Runtime-global capabilities apply to every root and child Agent. If a capability should apply only to one Agent handle, pass it to `runtime.agent(..., capabilities=...)` instead.

The effective definition digest includes the Agent declaration, model binding, output binding, runtime-global capabilities, local durable capabilities, active provider semantics, root Agent catalog source, and platform execution policy. A durable restore recomputes that identity and fails closed if the current runtime cannot reproduce it exactly.

## 5. Output

Output is a runtime binding, not part of `AgentSpec` and not a workspace-global registry:

```python
from pydantic import BaseModel

class Finding(BaseModel):
    title: str
    severity: str

result = await runtime.agent("audit", output=Finding).run("inspect the patch")
```

Durable state stores the Python output type module/qualname plus schema identity. Recovery imports the same public type and verifies the schema fingerprint before executing.

## 6. Sessions and history

```python
agent = runtime.agent("audit")
session = await agent.create_session("chat-1")
result = await agent.run("hello", session_id=session.session_id)

page = await runtime.session.history(
    session.session_id,
    principal=runtime.default_principal,
    cursor=None,
    limit=100,
)
```

A Session is pinned to one compiled Agent definition. Reusing a Session with a different effective definition is rejected with `SESSION_BINDING_MISMATCH`.

Session history is the committed conversation projection. Active, failed, and cancelled turns are not committed as successful conversation history. Execution trace/transcript remain separate audit views.

## 7. Runtime persistence

`RuntimeStatePlan` selects persistence per Runtime domain. The workspace default stores filesystem state under `<project>/.linktools/runtime`.

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

## 8. Durable execution identity

New executions persist an `AgentBindingSnapshot` in both `ExecutionRecord` and durable recovery input. It contains only data required to restore the effective local definition:

- canonical `AgentSpec`
- output Python type and schema identity
- ordered local Runtime capability descriptors
- effective binding digest

Current writers always persist the snapshot. Frozen legacy Runtime State V1 records without planning/thinking/snapshot remain readable as legacy data; partially upgraded shapes are rejected.

Retry and fork restore the source definition and inherit source planning/thinking. Recovery requires Execution and Recovery snapshots/modes to agree exactly.

## 9. TaskGraph and Temporal

`AgentHandle.task()` writes the final TaskGraph Agent V1 input with:

```text
agent_id
binding_digest
binding snapshot
user_prompt
planning
thinking
```

The Runtime admission boundary accepts the frozen legacy root-only V1 shape exactly, resolves it against the root Agent catalog, and normalizes it to the final V1 shape before persistence. Partial or unknown shapes are rejected. The Runtime Task planner consumes only the final V1 shape and restores the persisted snapshot before admitting an execution.

Temporal execution transport follows the same V1 rule. Current writers persist `planning`, `thinking`, `binding_digest`, and the full binding snapshot. The reader accepts either the frozen legacy V1 field set or the final V1 field set exactly; partial shapes are rejected. The workflow pins the binding digest and verifies it when loading the persisted request.

Applications integrating Temporal should use the public `WorkflowGateway` / worker components instead of parsing request objects directly.

## 10. Identity boundaries

| Field | Meaning |
|---|---|
| Runtime `namespace` | Isolates one Runtime data set inside a storage target |
| `tenant_id` | Authorization and resource ownership boundary |
| `memory_scope` | Selects a memory collection inside one tenant |
| Asset `namespace` | Isolates raw Asset storage; unrelated to Runtime state |
| Asset `kind` | Logical Asset type such as `agent`, `skill`, or `mcp` |
| Task/Tool `owner` | Current lease holder |

`open_workspace_runtime()` uses `workspace.workspace_id` as Runtime namespace. Its optional `tenant_id` defaults to `default`.

## 11. Development checks

Install the editable development environment once, then use the repository gate:

```bash
python manage.py install --editable
python manage.py check linktools-ai
```

AI-specific release checks and evidence live under root `scripts/check/ai`; `linktools-ai` does not maintain a package-local release-script tree.
