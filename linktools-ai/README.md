# linktools-ai

`linktools-ai` is the agent runtime for local workspace execution and durable
service integrations. Runtime topology is selected by the concrete launcher,
workflow gateway and persistence dependencies supplied by the application; the
runtime does not expose an execution profile or deployment category.

Its public storage abstraction is `AssetStore`; specification DTOs and codecs
live in `linktools.ai.spec`, while the concrete SQL asset backend is available
as `linktools.ai.asset.SqlAssetBackend`.

Storage builders are lazy. Callers explicitly initialize the SQL schema and
then construct the storage composition and asset store. The `ai run` command
streams model text, thinking and tool activity; `ai acp` serves the local ACP
transport when its optional dependency is installed.

For local execution, `--project` selects the working directory and `--storage`
selects the Runtime state directory:

```bash
ai-run --project /workspace/project --storage /var/lib/linktools-ai "hello"
ai-acp --project /workspace/project --storage /var/lib/linktools-ai
```

Sessions and execution records are stored below `<storage>/.linktools/`;
tools and agent files remain rooted at the project directory.

For database-backed downstream services, use the public store configuration and
keep the namespace stable for the deployment:

```python
from linktools.ai import RuntimePersistenceConfig, open_runtime_resources

config = RuntimePersistenceConfig.postgresql(
    namespace="tenant-id",
    deployment_id="runtime-prod",
)

# engine and session_factory are injected by the application composition.
async with open_runtime_resources(
    config,
    engine=engine,
    session_factory=session_factory,
) as resources:
    # Pass resources.domain into the application service composition.
    await resources.domain.sessions.list(tenant_id="tenant-id")
```

Use `RuntimePersistenceConfig.sqlite(path, namespace=..., deployment_id=...)` for a
single-node database. MySQL and PostgreSQL receive their injected engine and session
factory when `open_runtime_resources()` is called. SQL drivers are loaded only when
the corresponding backend is opened.

The checked-in architecture and release contract is maintained under
`scripts/build/matrix`. For profile-removal upgrades, drain unfinished affected
start/create requests and snapshot FILE/SQL durable state first. If rollback to
the old binary is required after a durable mutation, restore that snapshot
before rollback; no migration framework is introduced.

## Non-workspace pipeline execution

`open_runtime_resources()` + `build_local_runtime_services()` +
`RuntimeDependencies` + `build_runtime()` + `Runtime.run()` is the supported
process-local API for workers and pipelines that do not use a Workspace root. Compose
the resources and dependencies once when the process starts, then build a Runtime for
each job. The same dependencies can serve multiple `AgentSpec` and `PromptSpec`
variants.

`RuntimePersistenceConfig.in_memory(...)` keeps the runtime ledger and execution
history only for the current process. It does not recover runtime state after a
restart; downstream workers should use their own job checkpoint and audit storage for
that purpose. Use the existing SQLite or SQL persistence backends when linktools
execution/session/history must survive a restart.

The model example uses an OpenAI route and can point `base_url` at an
OpenAI-compatible endpoint. Other providers require a custom public `ModelMaterializer`.
Keep API keys in the credential provider, never in an AgentSpec, PromptSpec or metadata.

### Process startup

```python
import os

from pydantic import BaseModel

from linktools.ai.agent import OutputTypeRegistry
from linktools.ai.app import (
    RuntimeDependencies,
    RuntimePersistenceConfig,
    build_local_runtime_services,
    build_runtime,
    open_runtime_resources,
)
from linktools.ai.core import TenantAuthorizationPolicy, service_principal
from linktools.ai.model import (
    ModelConnectionConfig,
    ModelConnectionRegistry,
    ModelRegistry,
    ModelRoute,
    OpenAIModelMaterializer,
    SnapshotModelResolver,
    StaticModelCredentialProvider,
)
from linktools.ai.observe import MiddlewarePipeline
from linktools.ai.runtime import AllowAllToolPolicy, ExecutionRequest
from linktools.ai.spec import AgentSpec, PromptSpec
from linktools.ai.workspace import DisabledSandbox


class Answer(BaseModel):
    value: str


async def process_jobs() -> None:
    config = RuntimePersistenceConfig.in_memory(namespace="sec-smartops-worker")
    authorization = TenantAuthorizationPolicy()
    principal = service_principal(
        tenant_id="sec-smartops",
        principal_id="sec-smartops-worker",
    )
    credentials = StaticModelCredentialProvider(
        {"openai": os.environ["OPENAI_API_KEY"]}
    )
    materializer = OpenAIModelMaterializer(credentials)
    route = ModelRoute(
        route_id="default",
        provider="openai",
        model=os.environ["OPENAI_MODEL"],
        connection_id="default-openai",
    )
    model_registry = ModelRegistry()
    model_snapshot = model_registry.prime({"default": route})
    model_connections = ModelConnectionRegistry(
        (
            ModelConnectionConfig(
                connection_id="default-openai",
                base_url=os.getenv("OPENAI_BASE_URL"),
                credential_id="openai",
            ),
        )
    )
    output_types = OutputTypeRegistry()
    output_types.register("answer", 1, Answer)
    output_types.freeze()

    async with open_runtime_resources(config) as resources:
        # Process startup: compose this bundle once and retain it for all jobs.
        local = build_local_runtime_services(
            resources,
            authorization,
            grant_key=os.urandom(32),
            materializer=materializer,
        )
        dependencies = RuntimeDependencies(
            model_resolver=SnapshotModelResolver(model_snapshot),
            capability_resolvers=(),
            model_connections=model_connections,
            middleware=MiddlewarePipeline(),
            sandbox=DisabledSandbox(),
            tool_policy=AllowAllToolPolicy(),
            output_types=output_types,
            local=local,
        )

        # Job execution: choose declarations per job, while reusing dependencies.
        agent = AgentSpec(
            id="pipeline-agent",
            revision=1,
            model="default",
            capabilities=(),
            output_schema="answer",
            output_schema_revision=1,
            instructions=(),
        )
        prompt = PromptSpec(
            id="pipeline-prompt",
            revision=1,
            system="You are a structured pipeline agent.",
            instructions=(),
        )
        runtime = build_runtime(agent, prompt, dependencies=dependencies)
        result = await runtime.run(
            ExecutionRequest(
                prompt="process this job",
                principal=principal,
                idempotency_key="job-1",
            )
        )
        answer = Answer.model_validate(result.output)
        history = await runtime.execution.history(
            result.execution_id,
            principal=principal,
        )
        assert answer.value
        assert history.items

        # Repeated calls build another facade with the same process composition.
        repeated_runtime = build_runtime(agent, prompt, dependencies=dependencies)
        repeated_result = await repeated_runtime.run(
            ExecutionRequest("process another job", principal, idempotency_key="job-2")
        )
        variant_runtime = build_runtime(
            AgentSpec("pipeline-agent-variant", 1, "default", (), "answer", 1),
            PromptSpec("pipeline-prompt-variant", 1, prompt.system, ()),
            dependencies=dependencies,
        )
        variant_result = await variant_runtime.run(
            ExecutionRequest("process a variant", principal, idempotency_key="job-3")
        )
        Answer.model_validate(repeated_result.output)
        Answer.model_validate(variant_result.output)

        # Process shutdown: leaving this context closes the in-memory resources.
```

### Job execution

The first `build_runtime()` and `runtime.run()` in the sample execute one job, query
its process-local history, and validate structured output. The repeated call reuses
the same `dependencies`; the variant creates a different binding from new
declarations. A worker can place this block inside its asyncio job-consumer loop
without reopening resources for each job. The sample omits `memory_namespace` from
`ExecutionRequest`, so it remains `None` and durable memory is disabled.

### Process shutdown

The `open_runtime_resources()` context owns shutdown. Leaving it closes the in-memory
runtime resources after the worker has stopped accepting jobs. No runtime state is
promised after the process exits.

`DisabledSandbox()` is an explicit sentinel for a deployment without filesystem or
process sandbox capability. Its file and command operations always fail with
`SANDBOX_UNAVAILABLE` and have no side effects. `AllowAllToolPolicy()` is an explicit
permissive tool-policy choice; use it only when the deployment intentionally does not
require tool-level approval or filtering. It does not bypass the runtime
`AuthorizationPolicy`, remote capability permissions, or operating-system policy.

`AgentSpec` and `PromptSpec` are independent declarations that jointly participate in
binding. `AgentSpec` owns agent identity, model, capabilities, output schema,
instructions and metadata. `PromptSpec` owns prompt identity, system text and prompt
instructions. A single Agent can therefore be bound with multiple prompt variants;
the same instruction must not be duplicated in both declarations.

### Per-bind capability extensions

`capability_injections` adds a Pydantic AI capability to one binding. The AgentSpec
does not need a matching reference, and the injection identity changes the binding
digest. It is additive only: it cannot delete, replace or narrow a capability already
declared by the AgentSpec.

`additional_capability_resolvers` adds resolver implementations to one bind. The
AgentSpec must contain a matching provider reference for the resolver to contribute a
capability; an unused resolver contributes nothing. A provider already present in the
base resolver set is a `CAPABILITY_CONFLICT` and is never overridden.

```python
from linktools.ai.capability import CapabilityInjection

runtime = build_runtime(
    base_agent,
    prompt,
    dependencies=dependencies,
    capability_injections=(
        CapabilityInjection(
            id="temporary-tool",
            fingerprint=temporary_capability_fingerprint,
            capability=temporary_capability,
        ),
    ),
    additional_capability_resolvers=(SecSmartOpsCapabilityResolver(...),),
)
```

The injection fingerprint must represent all stable execution semantics of the
capability. Do not hide request IDs, job payloads, changing authorization outcomes,
or secrets in a capability object while reusing one fingerprint; request-scoped data
belongs in the existing request/run context.

Use declaration variants for subtractive or replacement behavior:

| Per-call intent | Supported form |
| --- | --- |
| Add one Pydantic capability | `capability_injections` |
| Add one provider resolver | `additional_capability_resolvers` |
| Narrow or replace capabilities | An `AgentSpec` variant |
| Change the model route | An `AgentSpec` variant |
| Change prompt instructions | A `PromptSpec` variant |
| Delete or override an existing capability/resolver | Not supported; use a new declaration/provider identity |

Custom providers should implement the public resolver contract only when a new
provider namespace is needed. Built-in Skill and MCP providers should normally reuse
`SkillCapabilityResolver` and `MCPServerCapabilityResolver`.

```python
from typing import Protocol

from linktools.ai.capability import CapabilityBinding
from linktools.ai.spec import AgentCapabilityRef


class CapabilityResolver(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...

    def resolve(
        self,
        refs: tuple[AgentCapabilityRef, ...],
    ) -> CapabilityBinding: ...
```
