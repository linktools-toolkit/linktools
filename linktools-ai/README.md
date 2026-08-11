# linktools-ai

`linktools-ai` separates raw Asset files, typed declarations and executable
runtime bindings:

```text
AssetStore -> AssetRepository -> AgentBindingComposer
           -> AgentBinder -> AgentBinding -> Runtime
```

`AssetStore` is the raw `AssetKey`-to-bytes boundary. `AssetRepository` owns
typed logical declaration loading and layout selection. `spec` owns declaration
DTOs and codecs; `capability` owns immutable capability bindings; `agent` owns
the synchronous binding compiler; and `app` is the only cross-domain
composition root.

## Asset-backed binding

Build an initialized raw store, then compose the typed repository and binding
composer. Agent and Prompt declarations are loaded through the repository;
runtime receives only the resulting frozen `AgentBinding`.

```python
from linktools.ai.agent import AssistantTextOutput, OutputTypeRegistry
from linktools.ai.app import (
    BOUND_RUNTIME_PROFILE_FINGERPRINT,
    build_agent_binding_composer,
    build_asset_repository,
    build_runtime,
)

assets = build_asset_repository(asset_store)
output_types = OutputTypeRegistry()
output_types.register("assistant-text", 1, AssistantTextOutput)
output_types.freeze()
composer = build_agent_binding_composer(
    assets,
    model_resolver=model_resolver,
    model_connections=model_connections,
    output_types=output_types,
    execution_profile_fingerprint=BOUND_RUNTIME_PROFILE_FINGERPRINT,
)
binding = await composer.compose(agent_id="coding", prompt_id="review")
runtime = build_runtime(binding, local=local_runtime_services)
result = await runtime.run(request)
```

`CapabilityPreparer` is the sole downstream capability extension point:

```python
class FooPreparer:
    provider = "foo"

    async def prepare(self, refs):
        return foo_binding(refs)

composer = build_agent_binding_composer(
    assets,
    model_resolver=model_resolver,
    model_connections=model_connections,
    output_types=output_types,
    execution_profile_fingerprint=BOUND_RUNTIME_PROFILE_FINGERPRINT,
    extra_preparers=(FooPreparer(),),
)
```

The composer stores an immutable provider mapping. Duplicate providers fail at
composition time. `AgentBinder` does not perform Asset I/O, provider lookup or
async work. Updating an Asset after composition does not change an existing
binding; a later composition creates a new snapshot.

Skill declarations are loaded as `SkillSpec` snapshots. A
`SkillCapabilityBinding` holds a `SkillCatalogSnapshot`, never an
`AssetStore`, `AssetRepository` or live Asset scope. MCP declarations are also
loaded from `AssetRepository`, while connections and toolsets are supplied by
`MCPRuntimeProvider` during execution.

## Local command

`lt ai run` uses a read-only local Asset root at
`<project>/.linktools` by default. Runtime persistence is stored at
`<project>/.linktools/runtime`; the workspace storage root remains the project
directory. The command loads the configured default Agent and Prompt assets,
falling back to read-only built-in defaults when they are absent. Skill
directories under `.linktools/skills` are exposed as `skill` Assets. It uses
the default deployment route and performs one-shot `Runtime.run()` execution.

```bash
lt ai run \
  --project /workspace/project \
  --model gpt-4o-mini \
  "review this change"
```

`--json` emits one final result object. Successful canonical output uses the
explicitly registered `assistant-text@1` schema and has
`ExecutionResult.output == {"text": "..."}`. The command does not use
`WorkspaceAgentRunner`, a live Agent catalog or `open_workspace_runtime`.

`--project` is also passed as the explicit `execution_root` to local runtime
services. The process working directory is never changed to select the
workspace root. The durable memory namespace is the workspace id.

## Runtime persistence

Storage builders are lazy. SQL runtime startup validates a pre-provisioned
schema and never creates or alters tables. Deployment platforms must provision
Runtime and StepStore tables before starting a SQL runtime. For an explicit
provisioning step, call `linktools.ai.migrate.provision_database(engine)` from
the application; the package does not expose a CLI for this operation.

For database-backed downstream services, inject the external session factory:

```python
from linktools.ai import RuntimePersistenceConfig, open_runtime_resources

config = RuntimePersistenceConfig.postgresql(
    namespace="tenant-id",
    deployment_id="runtime-prod",
)
async with open_runtime_resources(config, session_factory=session_factory) as resources:
    await resources.domain.sessions.list(tenant_id="tenant-id")
```

Use `RuntimePersistenceConfig.sqlite(...)` for a single-node database and keep
the namespace stable for the deployment. Runtime persistence and Asset storage
remain separate composition concerns.
