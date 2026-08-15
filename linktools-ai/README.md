# linktools-ai

`linktools-ai` loads workspace declarations, compiles immutable Agent definitions, and runs them through one local or durable Runtime:

```text
AssetStore -> AssetRepository -> AgentCompiler -> AgentDefinition -> Runtime
```

`AssetStore` owns raw files. `AssetRepository` resolves typed Agent, Skill, and MCP declarations. `AgentCompiler` produces an `AgentDefinition`; Runtime services execute it without adding provider-specific branches.

## 1. Run a workspace

### 1.1 Command line

Run from an installed command or the unified LinkTools entry point:

```bash
ai-run "review this change" --project /workspace/project --model gpt-4o-mini
python3 -m linktools ai run "review this change" --project /workspace/project --model gpt-4o-mini
```

`--base-url`, `--api-key`, and `--model` also read `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`. Add `--json` to emit one final result instead of streaming text and tool events.

### 1.2 Python API

```python
from linktools.ai import Workspace, open_workspace_runtime
from linktools.ai.model import ModelRegistry
from linktools.ai.workspace import trusted_workspace_principal

workspace = Workspace.discover("/workspace/project")
principal = trusted_workspace_principal(workspace.workspace_id)
models = ModelRegistry.openai(model="gpt-4o-mini")

async with open_workspace_runtime(workspace, models=models) as runtime:
    result = await runtime.run(
        "review this change",
        principal=principal,
        memory_scope=workspace.workspace_id,
    )
```

The same Runtime supports durable sessions and streaming:

```python
async with open_workspace_runtime(workspace, models=models) as runtime:
    session = await runtime.create_session("chat-1", principal=principal)
    async for event in runtime.stream(
        "hello",
        principal=principal,
        session_id=session.session_id,
        memory_scope=workspace.workspace_id,
    ):
        print(event.event_type, event.payload)
```

Named Agents use one handle:

```python
result = await runtime.agent("coding").run("review this change", principal=principal)
```

## 2. Asset loading

### 2.1 Default workspace loader

`ai-run` and `open_workspace_runtime()` read declarations from `<project>/.linktools/assets` through `Workspace.storage_root`. The default Asset source is a read-only `DirectoryAssetBackend`, while generated defaults use the selected Asset writer.

```text
.linktools/assets/
├── agents/<id>
├── integrations/mcp/<id>
└── capabilities/skills/<id>/SKILL.md
```

Agent and MCP declarations use their JSON codecs even though the canonical filenames have no suffix. Skills default to the directory layout shown above and may also use the single-file JSON representation.

Workspace startup resolves capabilities twice. The first pass discovers built-in and application providers; the second builds the final default Agent from the resulting capability references. Startup fails if the two passes disagree.

### 2.2 Custom logical Asset types

Pass `AssetTypeBinding` values and capability providers directly to the workspace composition:

```python
async with open_workspace_runtime(
    workspace,
    asset_bindings=(my_binding,),
    capability_providers=(my_provider,),
) as runtime:
    result = await runtime.run("use the custom capability", principal=principal)
```

Bindings are frozen before loading starts. Register every variant, codec, identity rule, and default write representation during composition.

### 2.3 Custom Asset storage

The workspace helper intentionally owns its local-directory loader. Applications that need a database, remote service, or another loading policy compose the public Asset boundary directly:

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

`MyAssetBackend` should implement the public `AssetBackend` and storage contracts. Use `StorageLayer` to add ordered fallback sources. Do not import another package's private module or call private backend methods.

Built-in backends include `InMemoryAssetBackend`, `DirectoryAssetBackend`, `FilesystemAssetBackend`, and `SqlAssetBackend`. A SQL-backed store receives an `AsyncEngine`:

```python
from linktools.ai.asset import AssetStore, SqlAssetBackend
from linktools.ai.storage import StorageOverlay
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine("sqlite+aiosqlite:///assets.db")
backend = SqlAssetBackend(engine, namespace="assets")
store = AssetStore(StorageOverlay(backend, writer=backend))
await store.initialize()
```

The SQL schema must already exist. `AssetStore.initialize()` initializes its overlay, which calls `initialize()` once on each backend and validates the SQL schema before reads or writes.

## 3. Runtime persistence

`RuntimeStatePlan` selects the runtime persistence routes. Filesystem persistence is the workspace default and writes below `<project>/.linktools/runtime`.

Unselected domains use process-local stores and are intentionally absent after restart.

The identity fields are intentionally independent:

| Field | Meaning |
|---|---|
| Runtime `namespace` | The workspace identity that isolates one Runtime data set inside a target |
| `tenant_id` | Authorization and resource ownership boundary inside that data set |
| `memory_scope` | Selects a memory collection inside one tenant |
| Asset `namespace` | Isolates raw Asset data and is unrelated to Runtime storage |
| Asset `kind` | Selects a logical Asset type such as `agent`, `skill`, or `mcp` |
| Task/Tool `owner` | Identifies the current lease holder |

Fields stay short when their declaring type supplies the domain, such as `Principal.kind` and `TaskLease.owner`. Explicit qualifiers distinguish meanings that can coexist in the same record or flattened storage boundary, such as `operation_kind`, `resource_kind`, `lineage_kind`, `asset_kind`, and `memory_scope_key`, or preserve an authorization identity domain, such as `owner_principal_id`.

`open_workspace_runtime()` uses `workspace.workspace_id` as both the Runtime namespace and the Workspace tenant boundary. Lower-level domain stores remain multi-tenant through their explicit `tenant_id` fields.

External SQL deployments borrow an application-owned async SQLAlchemy `AsyncEngine` and require a pre-provisioned schema. Runtime startup validates tables; explicit migration provisioning adds only the supported task columns:

```python
from linktools.ai.runtime import RuntimeState, RuntimeStatePlan, RuntimeStateRoute
from linktools.ai.model import ModelRegistry
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine("sqlite+aiosqlite:////var/lib/my-app/runtime.db")
models = ModelRegistry.openai(model="gpt-4o-mini")
runtime_state = RuntimeState.from_plan(
    RuntimeStatePlan(
        conversation=RuntimeStateRoute.sql(engine),
        memory=RuntimeStateRoute.sql(engine),
    )
)
async with open_workspace_runtime(workspace, runtime=runtime_state, models=models) as runtime:
    result = await runtime.run("inspect the patch", principal=principal)

await engine.dispose()
```

MySQL, PostgreSQL, and SQLite use the same SQL storage contracts. Every lifecycle object uses `initialize()`.

The application owns an external engine and must dispose it after the Runtime closes.

A non-empty `memory_scope` enables Runtime memory for an execution. `None` disables it. Classification fields reject empty values, surrounding whitespace, control characters, and values beyond their UTF-8 byte limit.

## 4. Capability providers

Applications add providers when composing the compiler. Providers resolve declarations through `AssetRepository` and return immutable `CapabilityBinding` values:

```python
compiler = AgentCompiler(
    assets,
    model_resolver=model_resolver,
    output_types=output_types,
    capability_providers=(my_provider,),
    capability_grants=workspace_grants,
    execution_profile_fingerprint=profile_digest,
)
```

Runtime and `AgentExecutor` remain independent of provider implementations.

## 5. Standalone Task API

Task scheduling can run without constructing Runtime, Agent, Model, or Workspace objects:

```python
from linktools.ai.task import open_local_task_api

async with open_local_task_api(
    persistence,
    authorization,
    runner=my_task_runner,
    owner="worker-1",
) as tasks:
    result = await tasks.run_graph(request)
```

The runner implements `TaskNodeRunner` and returns a canonical `TaskNodeRunResult`. The local launcher owns leases and heartbeats; durable task state stays in the supplied persistence.

## 6. Development checks

From the repository root:

```bash
PYTHONPATH=linktools-ai:linktools-ai/src:linktools/src python3 -m compileall -q linktools-ai/src/linktools/ai
PYTHONPATH=linktools-ai:linktools-ai/src:linktools/src python3 -m pytest -q tests/ai
PYTHONPATH=linktools-ai:linktools-ai/src:linktools/src python3 -m ruff check linktools-ai/scripts/build linktools-ai/src/linktools/ai linktools-ai/src/linktools/commands/ai tests/ai
```
