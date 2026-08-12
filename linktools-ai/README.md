# linktools-ai

`linktools-ai` loads workspace declarations, compiles immutable Agent definitions, and runs them through one local or durable Runtime:

```text
AssetStore -> AssetRepository -> AgentCompiler -> AgentDefinition -> Runtime
```

`AssetStore` owns raw files. `AssetRepository` resolves typed Agent, Prompt, Skill, and MCP declarations. `AgentCompiler` produces an `AgentDefinition`; Runtime services execute it without adding provider-specific branches.

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
from linktools.ai.workspace import trusted_workspace_principal

workspace = Workspace.discover("/workspace/project")
principal = trusted_workspace_principal(workspace.workspace_id)

async with open_workspace_runtime(workspace, model="gpt-4o-mini") as runtime:
    result = await runtime.run(
        "review this change",
        principal=principal,
        memory_namespace=workspace.workspace_id,
    )
```

The same Runtime supports durable sessions and streaming:

```python
async with open_workspace_runtime(workspace, model="gpt-4o-mini") as runtime:
    session = await runtime.create_session("chat-1", principal=principal)
    async for event in runtime.stream(
        "hello",
        principal=principal,
        session_id=session.session_id,
        memory_namespace=workspace.workspace_id,
    ):
        print(event.event_type, event.payload)
```

Compile a named Agent and Prompt with `runtime.compile_agent("coding", prompt_id="review")`, or pass those IDs to `runtime.run()`.

## 2. Asset loading

### 2.1 Default workspace loader

`ai-run` and `open_workspace_runtime()` read declarations from `<project>/.linktools`. The directory is a read-only `LocalDirectoryAssetBackend`; generated defaults live in a writable in-memory layer.

```text
.linktools/
├── agent/<id>
├── prompt/<id>
├── mcp/<id>
└── skills/<id>/SKILL.md
```

Agent, Prompt, and MCP declarations use their JSON codecs even though the canonical filenames have no suffix. Skills default to the directory layout shown above and may also use the single-file JSON representation.

Workspace startup resolves capabilities twice. The first pass discovers built-in and application providers; the second builds the final default Agent from the resulting capability references. Startup fails if the two passes disagree.

### 2.2 Custom logical Asset types

Pass `AssetTypeBinding` values through `extra_asset_bindings`. Capability providers that resolve those assets can be added with `capability_provider_factories`:

```python
async with open_workspace_runtime(
    workspace,
    model="gpt-4o-mini",
    extra_asset_bindings=(my_binding,),
    capability_provider_factories=(my_provider_factory,),
) as runtime:
    result = await runtime.run("use the custom capability", principal=principal)
```

Bindings are frozen before loading starts. Register every variant, codec, identity rule, and default write representation during composition.

### 2.3 Custom Asset storage

The workspace helper intentionally owns its local-directory loader. Applications that need a database, remote service, or another loading policy compose the public Asset boundary directly:

```python
from linktools.ai.asset import AssetStore
from linktools.ai.storage import StorageOverlay
from linktools.ai.workspace import build_workspace_asset_repository

backend = MyAssetBackend(...)
store = AssetStore(StorageOverlay(backend, writer=backend))
await store.initialize()
assets = build_workspace_asset_repository(store, extra_bindings=(my_binding,))
```

`MyAssetBackend` should implement the public `AssetBackend` and storage contracts. Use `StorageLayer` to add ordered fallback sources. Do not import another package's private module or call private backend methods.

Built-in backends include `InMemoryAssetBackend`, `LocalDirectoryAssetBackend`, `FilesystemAssetBackend`, and `SqlAssetBackend`. A SQL-backed store remains simple to construct:

```python
from linktools.ai.asset import AssetStore, SqlAssetBackend
from linktools.ai.storage import StorageOverlay

backend = SqlAssetBackend(session_factory, namespace="assets")
store = AssetStore(StorageOverlay(backend, writer=backend))
await store.initialize()
```

The SQL schema must already exist. `AssetStore.initialize()` initializes its overlay, which calls `initialize()` once on each backend and validates the SQL schema before reads or writes.

## 3. Runtime persistence

`RuntimePersistenceConfig.in_memory(...)`, `.filesystem(...)`, `.sqlite(...)`, `.mysql(...)`, and `.postgresql(...)` select the deployment. Filesystem persistence is the workspace default and writes below `<project>/.linktools/runtime`.

SQL deployments require an application-owned async SQLAlchemy session factory and a pre-provisioned schema. Runtime startup validates tables but never creates or alters them:

```python
from linktools.ai.migrate import provision_database
from linktools.ai.workspace import RuntimePersistenceConfig
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

path = "/var/lib/my-app/runtime.db"
config = RuntimePersistenceConfig.sqlite(
    path,
    namespace="project-a",
    deployment_id="local",
)
engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
session_factory = async_sessionmaker(engine, expire_on_commit=False)

await provision_database(engine)
async with open_workspace_runtime(
    workspace,
    config=config,
    model="gpt-4o-mini",
    session_factory=session_factory,
) as runtime:
    result = await runtime.run("inspect the patch", principal=principal)
```

MySQL and PostgreSQL use the same `session_factory` entry point. SQLite enables its connection policy in the storage layer and keeps Harness step persistence in a namespace-scoped sibling database. Every lifecycle object uses `initialize()`.

The application owns the engine and session factory and must dispose the engine after the Runtime closes.

A non-empty `memory_namespace` enables Runtime memory for an execution. `None` disables it, and an empty string is invalid.

## 4. Capability providers

Applications add providers when composing the compiler. Providers resolve declarations through `AssetRepository` and return immutable `CapabilityBinding` values:

```python
compiler = AgentCompiler(
    assets,
    model_resolver=model_resolver,
    model_connections=model_connections,
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
