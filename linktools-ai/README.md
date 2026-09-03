# linktools-ai

`linktools-ai` provides the Agent runtime layer for LinkTools. The public composition model is intentionally small:

```text
Workspace
    + CapabilityGroup(s)
    + ModelRegistry
    + RuntimeState
        -> Runtime.open(...)
        -> frozen capability/declaration candidates
        -> AgentCompiler
        -> AgentDefinition
        -> Runtime.agent(id)
        -> per-execution AgentBinding
        -> Agent / Session / Execution / Task / Evaluation / Recovery
```

The main ownership rules are:

- `Workspace` owns workspace identity, paths, policy, and sandbox configuration.
- `AssetStore` stores raw asset bytes. It does not interpret declarations.
- `CapabilityGroup` is the only public registration/discovery composition unit. A group freezes direct registrations and, when store-backed, one immutable `AssetStore` snapshot.
- `AgentSpec` is a runtime-independent Agent declaration.
- `AgentCompiler` is the sole Agent-level selector. It resolves model, tool, Skill, MCP, capability, and Subagent candidates from the frozen Runtime candidate set.
- `Runtime` is the composition root and owns the service graph.
- `Runtime.agent(id)` returns a Runtime-bound `Agent`; it does not compile or register new definitions.
- `AgentBinding` is created per execution and pins the exact durable semantics, including the output contract.
- `Session` is bound to `AgentSpec.id`; retry/recovery remain pinned to the exact historical execution binding.

## 1. Run a workspace

### Command line

```bash
ai-run "review this change" --project /workspace/project --model gpt-4o-mini
python3 -m linktools ai run "review this change" --project /workspace/project --model gpt-4o-mini
```

Useful options:

- `--base-url`, `--api-key`, and `--model` also read `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.
- `--storage filesystem|sqlite` selects Runtime state storage.
- `--planning` enables planning for the execution.
- `--thinking` requests model thinking when supported.
- `--json` emits one terminal JSON result.

### Python

```python
from linktools.ai import Runtime, Workspace
from linktools.ai.model import ModelRegistry

workspace = Workspace.discover("/workspace/project")
models = ModelRegistry.openai(model="gpt-4o-mini")

async with Runtime.open(workspace, models=models) as runtime:
    result = await runtime.agent("default").run(
        "review this change",
        memory_scope=workspace.workspace_id,
        planning=True,
    )
```

`Runtime.open()` is the public composition root. The Runtime is frozen for the lifetime of the context; registrations are completed before it opens.

## 2. Define application capabilities and Agents

Use `CapabilityGroup` for direct application registrations:

```python
from linktools.ai import CapabilityGroup, RunContext, Runtime

application = CapabilityGroup[None]("application")

@application.tool
def lookup_ticket(ctx: RunContext[None], ticket_id: str) -> str:
    return ticket_id

application.agent(
    "audit",
    model="default",
    system_prompt="Review the supplied evidence carefully.",
    allow_tools=("lookup_ticket",),
    allow_skills=("review",),
    allow_subagents=(),
)

async with Runtime.open(
    workspace,
    models=models,
    capabilities=(application,),
) as runtime:
    result = await runtime.agent("audit").run("inspect ticket SEC-123")
```

`CapabilityGroup.tool()` and `CapabilityGroup.capability()` accept a positive semantic `revision`. The revision is an explicit fingerprint input for Python behavior whose semantics cannot be reconstructed from a declaration payload. It is not a project-wide version layer.

`CapabilityGroup.agent()` creates an `AgentSpec`; declarations themselves use the single v1 wire contract and do not expose a per-declaration revision field.

## 3. Workspace declarations

When no group named `workspace` is supplied, `Runtime.open()` creates the standard workspace source from `.linktools` and loads these built-in declaration kinds:

```text
.linktools/
  agents/<id>
  skills/<id>
  skills/<id>/SKILL.md
  mcp/<id>
```

The default workspace source is a raw `AssetStore`. `CapabilityGroup.from_store()` performs declaration discovery over one immutable store snapshot:

```python
from linktools.ai import CapabilityGroup, Runtime
from linktools.ai.asset import AssetStore

workspace_group = CapabilityGroup.from_store("workspace", my_asset_store)

async with Runtime.open(
    workspace,
    models=models,
    capabilities=(workspace_group,),
) as runtime:
    ...
```

A store-backed group reads metadata, batch-loads the corresponding bytes, verifies content identity, runs its loaders, and verifies that the store revision did not change during the freeze. Conflicting identities or layouts fail closed.

For downstream declaration formats or custom kinds such as `worker` or `audit`, implement `CapabilityLoader` and register it with `group.loader(loader)`. The loader receives the frozen `AssetInfo` sequence and matching byte mapping and returns normal `CapabilityContribution` values. No additional Registry/Provider abstraction is required.

### Workspace sandbox

Workspace filesystem and shell tool effects run through the public `Sandbox` / `SandboxSession` boundary. Inject a custom implementation with `Workspace(..., sandbox=...)`, `Workspace.load(..., sandbox=...)`, or `Workspace.discover(..., sandbox=...)`.

When `sandbox=None`, LinkTools uses its built-in local adapter. That adapter delegates actual filesystem/process operations to the local Harness implementation, but LinkTools owns the stable model-visible workspace tool signatures, descriptions, metadata, and durable semantic pins.

A run with no selected workspace filesystem/shell tools does not open a sandbox. Otherwise the run opens exactly one `SandboxSession`; filesystem tools, foreground shell commands, and background `start/check/stop` commands share that session, which is closed when the model run succeeds, fails, or is cancelled.

Use `DisabledSandbox` to keep workspace tool declarations and historical binding recovery available while making runtime workspace tool materialization fail with `SANDBOX_UNAVAILABLE`. A custom Sandbox failure does not fall back to the local host environment.

Sandbox v1 virtualizes only workspace filesystem and shell tool effects. Runtime state, `AssetStore`, Skill loading, and repository-instruction discovery are not automatically moved into a remote Sandbox. A remote implementation must therefore expose the intended logical project tree itself; LinkTools does not provide project-tree synchronization for an unsynchronized remote Sandbox in v1.

The built-in local Sandbox is an execution boundary, not a claim of container- or VM-level operating-system isolation.

## 4. Agent selection and capability policy

`AgentSpec` contains declarative selection policy:

```python
from linktools.ai.spec import AgentSpec

spec = AgentSpec(
    id="audit",
    model="default",
    system_prompt="Audit the supplied change.",
    instructions=("Cite concrete evidence.",),
    allow_tools=("read_file", "mcp__security__*"),
    allow_skills=("review",),
    allow_subagents=("triage",),
    planning=True,
    thinking="high",
)
```

The compiler resolves these selectors once from the frozen candidate universe. Missing or conflicting required candidates fail closed.

`allow_tools` controls ordinary/external model-visible tools. Planning is an execution mode and is not enabled or disabled by pretending `write_plan` is an ordinary business tool. Runtime infrastructure capabilities such as planning, memory, Skill loading, and Subagent delegation are composed by Runtime according to the resolved execution contract.

Subagents are root Agent definitions selected from the same frozen catalog. A root Agent cannot select itself as a Subagent, and the Runtime does not create a second registration system for child Agents.

## 5. Output contracts

Output belongs to an execution, not to `AgentSpec`, `Runtime.agent()`, or Session identity:

```python
from pydantic import BaseModel

class Finding(BaseModel):
    title: str
    severity: str

agent = runtime.agent("audit")
result = await agent.run(
    "inspect the patch",
    output=Finding,
)
```

The exact durable binding stores:

- the v1 `AgentSpec` semantic payload;
- the resolved model semantic payload;
- the selected semantic pins;
- selected Subagent ids;
- `output_mode`;
- the canonical output JSON Schema;
- one `binding_digest`.

The snapshot does not persist Python output import paths, duplicate output schema ids/revisions/fingerprints, or a second binding fingerprint. `ExecutionResult` exposes the derived `output_fingerprint` together with the terminal output.

## 6. Sessions and executions

```python
agent = runtime.agent("audit")
session = await agent.create_session("chat-1")

first = await session.run("inspect the first change")
second = await session.run(
    "return a structured summary",
    output=Finding,
    planning=True,
)

history = await session.history()
```

A Session owns conversation continuity and the stable Agent id. Every new execution binds the current frozen Agent definition to that execution's output contract. Retry, fork, durable recovery, evaluation, and Task execution use the exact binding snapshot/digest required by their contract rather than re-running current selector discovery.

User prompt transport is also durable: plain text uses the `text` codec, while supported native Pydantic user content uses the v1 durable user-content codec. Unsupported external file lifecycle objects fail closed instead of being guessed or silently converted.

## 7. Runtime state

`Runtime.open()` accepts an explicit `RuntimeState` when the application owns storage selection:

```python
from linktools.ai import Runtime
from linktools.ai.runtime import RuntimeState

state = RuntimeState.sqlite("/var/lib/linktools/runtime.db")

async with Runtime.open(
    workspace,
    models=models,
    state=state,
) as runtime:
    ...
```

Built-in Runtime state supports in-memory, filesystem, SQLite, and SQL composition used by the Runtime persistence layer. State domains keep their existing ownership, transaction, recovery, and retention rules; `Runtime.open()` consumes the state object instead of exposing duplicate storage-root arguments.

SQLite-backed Runtime state supports the built-in durable TaskGraph scheduler without a SQLite-specific launcher or an external lock. Normal internal Task optimistic-CAS races are reread and converged by the Task domain. Durable ToolOperation terminal persistence is also lease-aware: a same-lease heartbeat racing terminal persistence is reconciled without replaying the tool effect. Genuine ownership, fence, idempotency, tool-result, effect-unknown, integrity, and storage errors remain observable. Runtime startup still does not provision or migrate database schemas; schema provisioning remains an explicit deployment step.

Durable local execution and recovery are provided by Runtime state and recovery checkpoints and do not require an external workflow server.

## 8. Execution failure diagnostics

`execution-error-diagnostics-v1` extends failed execution results with durable diagnostic context while keeping the existing safe error contract unchanged:

```python
result = await execution.wait()

result.error_code
result.safe_error_details
result.error_diagnostics
```

`error_code` remains the stable machine-readable classification used by Runtime control flow. `safe_error_details` remains the redacted/safe mapping for ordinary application handling. A failed execution may additionally expose `error_diagnostics` with:

- `exception_type`: original exception class name, at most 256 Unicode code points;
- `exception_message`: `str(original_exception)`, at most 2048 Unicode code points;
- `cause_digest`: 64 lowercase hexadecimal SHA-256 characters computed from the untruncated original exception type and message.

`exception_message` is diagnostic data, not a safe/redacted field. It may contain sensitive text already present in the originating exception. Runtime does not add prompts, model responses, headers, request payloads, or URLs to diagnostics, but it also does not redact the exception message. Applications should apply the same authorization and retention controls to diagnostics as to other execution investigation data.

Diagnostics are durable with the failed execution and are returned after Runtime restart through the normal public execution result and terminal event APIs. Historical failed records that predate the field return `error_diagnostics is None`. Successful and cancelled executions never carry diagnostics.

Diagnostics do not participate in error classification, retry decisions, TaskGraph scheduling, idempotency identity, or terminal-state decisions. The field meanings, truncation limits, and digest input above define `execution-error-diagnostics-v1`; incompatible semantic changes require a new diagnostics contract version.

The `ai run --json` terminal object includes `error_diagnostics`; ordinary non-JSON CLI failure text continues to report only `error_code` and `safe_error_details`.

## 9. Public API boundary

The top-level composition API is intentionally small:

```python
from linktools.ai import (
    Agent,
    CapabilityGroup,
    Execution,
    RunContext,
    Runtime,
    Session,
    Workspace,
)
```

Package-specific public contracts remain available from their owning packages, for example `linktools.ai.asset`, `linktools.ai.model`, `linktools.ai.spec`, and `linktools.ai.runtime`. `ErrorDiagnostics` is available from `linktools.ai.runtime`.

Private modules prefixed with `_` are implementation details. Downstream applications should not import Runtime execution infrastructure, state repository internals, or private compiler helpers directly.
