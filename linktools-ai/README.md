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

workspace = Workspace.initialize("/workspace/project")
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
from linktools.ai import AgentContext, CapabilityGroup, Runtime

application = CapabilityGroup[None]("application")

@application.tool
def lookup_ticket(ctx: AgentContext[None], ticket_id: str) -> str:
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

`CapabilityGroup.tool()` and `CapabilityGroup.capability()` accept a positive semantic `revision`. The revision is an explicit fingerprint input for Python behavior whose semantics cannot be reconstructed from a declaration payload. Generic Pydantic capabilities retain their native Pydantic AI behavior, including model selection, tools, lifecycle hooks, deferred loading, and per-agent/per-run binding. Their `semantic_id`, `revision`, and optional `semantic_config` are recorded in the Agent binding; `semantic_config` should be supplied when runtime configuration changes semantics without changing the revision. LinkTools reserves only its own internal capability identities and revalidates final output against the durable `OutputBinding`. It is not a project-wide version layer.

Generic capabilities are trusted host-Python extensions. LinkTools preserves their native hooks and does not sandbox or deny their file, network, or process access; only LinkTools-owned workspace, opaque-effect, and deferred-resolution boundaries provide those controls.

LinkTools-owned Tool declarations carry their runtime semantics in
`ToolDefinition.metadata`. `CapabilityGroup.tool()` writes the registered
`effect`, `plan_safe`, and `tool_class=business` values there; Workspace and
MCP owners declare their own effect, class, path, and context semantics. Plan
filtering, sandbox selection, leaf effect handling, and compaction consume
these declarations instead of inferring behavior from Tool names.

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

When `sandbox=None`, LinkTools uses its built-in local adapter. LinkTools owns
the stable model-visible workspace tool signatures, descriptions, metadata, and
durable semantic pins.

A run with no selected workspace filesystem/shell tools does not open a sandbox. Otherwise the run opens exactly one `SandboxSession`; filesystem tools, foreground shell commands, and background `start/check/stop` commands share that session, which is closed when the model run succeeds, fails, or is cancelled.

Use `DisabledSandbox` to keep workspace tool declarations and historical binding recovery available while making runtime workspace tool materialization fail with `SANDBOX_UNAVAILABLE`. A custom Sandbox failure does not fall back to the local host environment.

`LocalSandbox` runs with the workspace as its current directory and is an
execution boundary, not an operating-system security boundary. On Linux,
`BubblewrapSandbox` is an explicit deployment choice. It requires a non-root
user, usable unprivileged namespaces, `bwrap >= 0.12.0`, and a trusted
read-only runtime rootfs containing the same LinkTools build and Python >=
3.10. It has no automatic Local fallback. Bubblewrap isolates only Session
file/command execution; the host Agent, model requests, Python custom tools,
and MCP remain outside it. Its network namespace provides Session loopback and
no external route, but shared workspace files and explicitly shared IPC remain
outside that guarantee.

Selected local Skills are exposed as read-only `SandboxResource` directories at
`/skills/<key>` in Bubblewrap and at their validated host location in Local
sessions. The mapping is derived for the current run and is not persisted into
Skill declarations. Background command state is ephemeral and is cleaned up
with the Session; it is not a cross-run service.

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

URL and uploaded-file content remains an external reference: Runtime persists its declared metadata and does not implicitly download it. Inline binary content and workspace file inputs are frozen as bytes before execution reservation when their durable contract requires it. Model transport retries use `max_retries=2` with a fixed `retry_delay=1.0`; tool correction defaults to `tool_retries=10000`, and output correction defaults to `output_retries=3`.

OpenAI model declarations accept explicit `vision=True` or `vision=False`
(`False` by default). The value is part of the model semantic fingerprint. A
model with `vision=False` rejects recognizable image content at the final
request boundary before provider calls or transport retries; PDF, text, and
other non-image attachments continue through the existing provider contract.
LinkTools does not infer image support from model names, endpoints, or probes.

Execution file input uses the same durable boundary:

```python
result = await agent.run(
    "分析这些截图",
    files=("screenshots/overview.png", "screenshots/details.png"),
)
```

The Sandbox canonicalizes and deduplicates logical paths before reading them. The initial model request receives each file as `BinaryContent` together with its canonical Workspace path, and the captured bytes are recovered from Runtime state rather than reread from the Workspace during retry or recovery. After a complete model response consumes that binary input, Runtime keeps only lightweight file/path context in the active model context, so later agent-loop requests, Session turns, and forks do not repeatedly resend the bytes. The raw transcript remains lossless.

If an Agent needs to inspect a Workspace file again, select `attach_files` in `allow_tools`. `attach_files(paths=[...])` is a normal `filesystem.read` Workspace tool: it applies the existing Sandbox, path, approval, and repository-instruction boundaries, reads the current Workspace contents, and sends those files only to the next model request. A later complete model response consumes them under the same transient rule. `Agent.task()` remains a generic TaskGraph API and does not accept `files`; delegated subagents use the same explicit `files=` execution input.

### Runtime context and execution queries

`Runtime.open()` is the stable Runtime construction entry point. Use
`RuntimeContext` to provide the application object, tenant, correlation, and
low-cardinality metric dimensions for the Runtime lifetime:

```python
from linktools.ai.runtime import RuntimeContext

context = RuntimeContext(app, tenant_id="tenant-a")
async with Runtime.open(workspace, models=models, context=context) as runtime:
    ...
```

Execution metadata is available through the authorized public query surface:

```python
from linktools.ai.runtime import ListExecutionRequest

page = await runtime.execution.list(
    ListExecutionRequest(
        principal=runtime.default_principal,
        session_id="chat-1",
        agent_id="audit",
    )
)
```

`RuntimeHistory.open()` provides the same read-only execution metadata and
history projections without loading a ModelRegistry or compiling Agents.
`RuntimeHistory.inspect_execution()` and `list_executions()` use the same
authorization and filters as `Runtime.execution`. Execution list cursors are
HMAC-protected, ordered weak-consistency continuations: they do not provide a
cross-page MVCC snapshot, and a caller needing a closed set should query after
the target set is stable or restart from the first page.

Downstream code must not scan `ExecutionRecord`, codec data, `StateStore`, or
private repositories directly. No durable schema or data migration is part of
this Runtime query surface.

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

Durable local execution and recovery are provided by Runtime state and recovery
checkpoints and do not require an external workflow server. Harness provides the
Planning, Memory, StepPersistence, and context-compaction capability behavior,
while LinkTools remains the durable owner of plans, Memory records and mutation
receipts, execution history, and the raw transcript. Memory content continues to
persist through `MemoryState` and `ObjectStore`. Compaction only rewrites the
request context projection; it never rewrites the raw transcript.

### Workspace relocation

Use an explicit logical `workspace_id` when a Workspace must survive a physical move:

```python
workspace = Workspace.load(
    "/new/project",
    workspace_id="workspace-prod-01",
)
state = RuntimeState.filesystem("/new/runtime-state")
```

`Workspace.root`, Runtime state paths, SQLite paths, SQL endpoints, ObjectStore `store_id`, and storage topology are deployment details. Runtime persistence keeps logical Workspace paths and resolves object payloads by durable Runtime domain plus object key/digest/size, so `Runtime.open()` performs normal recovery after a consistent Workspace, state, and ObjectStore restore without freezing the original backend identity. A separate `Runtime.restore()` migration step is not required.

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
    AgentContext,
    CapabilityGroup,
    Execution,
    Runtime,
    Session,
    Workspace,
)
```

Package-specific public contracts remain available from their owning packages, for example `linktools.ai.asset`, `linktools.ai.model`, `linktools.ai.spec`, and `linktools.ai.runtime`. `ErrorDiagnostics` is available from `linktools.ai.errors`.

Private modules prefixed with `_` are implementation details. Downstream applications should not import Runtime execution infrastructure, state repository internals, or private compiler helpers directly.
