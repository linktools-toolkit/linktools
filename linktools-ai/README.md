# linktools-ai

`linktools-ai` keeps declaration loading, definition compilation, and execution
as separate boundaries:

```text
AssetStore -> AssetRepository -> AgentCompiler -> AgentDefinition -> Runtime
```

`AssetStore` is the raw file boundary. `AssetRepository` resolves typed Agent,
Prompt, Skill, and MCP declarations. `AgentCompiler` creates an immutable
`AgentDefinition`; `Runtime` executes any compiled definition through the same
durable services.

## One-shot execution

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

## Durable streaming chat

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

The same `Runtime` can compile and execute multiple Agent definitions:

```python
async with open_workspace_runtime(workspace, model="gpt-4o-mini") as runtime:
    definition = await runtime.compile_agent("coding", prompt_id="review")
    result = await runtime.run(
        "inspect the patch",
        principal=principal,
        agent_id="coding",
        prompt_id="review",
    )
```

## Capability provider

Applications add capability providers at compiler composition time. Providers
resolve declarations through `AssetRepository` and return immutable
`CapabilityBinding` values; Runtime and AgentExecutor do not contain
provider-specific branches.

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

## Workspace paths and persistence

The default workspace asset root is `<project>/.linktools`; runtime files are
stored under `<project>/.linktools/runtime`. Skill directories use the
canonical `skill/<id>/SKILL.md` Asset layout. The local directory layer is
read-only, so checked-in declarations can be overlaid by a writable backend.

`open_workspace_runtime()` accepts an external SQL `session_factory` for
pre-provisioned deployments. Runtime startup validates the schema and does not
create or alter production tables. Applications that own provisioning may call
`linktools.ai.migrate.provision_database(engine)` explicitly.

`RuntimePersistenceConfig.in_memory(...)`, `.filesystem(...)`, `.sqlite(...)`,
`.mysql(...)`, and `.postgresql(...)` select the persistence deployment. A
non-empty `memory_namespace` enables the Runtime memory capability; `None`
leaves memory disabled and an empty string is rejected.

## Standalone local Task API

Task scheduling can be composed without constructing Runtime, Agent, Model, or
Workspace objects:

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

`my_task_runner` implements `TaskNodeRunner` and returns only a canonical
`TaskNodeRunResult(result_digest, execution_id=None)`. The launcher owns local
leases and heartbeats; durable task state remains in the supplied persistence.

## Breaking cutover

The current decoder accepts only array allowlists. Before a coordinated cutover,
drain or cancel nonterminal work, verify that no old worker lease remains,
convert legacy boolean fields offline (`true` to `["*"]`, `false` to `[]`),
then validate every asset with the strict codec before starting the new workers.
Do not run old and new workers against the same workload. Rollback is another
coordinated cutover: drain new work, restore the pre-cutover AgentSpec backup,
stop all new workers, and start the complete old worker set only after leases
have expired or been cleared.
