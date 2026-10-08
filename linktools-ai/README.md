# linktools-ai

`linktools-ai` provides the Agent runtime layer for LinkTools. The public composition model is intentionally small:

```text
namespace
    + ModelRegistry
    + RuntimeStorage
    + CapabilityGroup(s)
        -> optional, independent Workspace and Sandbox via CapabilityGroup(...)
        -> Runtime.open(...)
        -> captured capability/declaration candidates
        -> AgentCompiler
        -> CompiledAgent
        -> runtime.agents.get(id)
        -> per-execution AgentBinding
        -> Agent / Session / Execution / Task / Evaluation / Recovery
```

The main ownership rules are:

- `Runtime` owns a stable persistence namespace; it does not require a filesystem Workspace.
- `Workspace` owns paths and policy; it is not a persistence identity.
- `AssetStore` stores raw asset bytes. It does not interpret declarations.
- `CapabilityGroup` is the only public registration/discovery composition unit. It can provide a Workspace or Sandbox independently and captures direct registrations and, when store-backed, one declaration view pinned to Asset version references.
- `AgentSpec` is a runtime-independent Agent declaration.
- `AgentCompiler` is the sole Agent-level selector. It resolves model, tool, Skill, MCP, capability, and Subagent candidates from the captured Runtime candidate set.
- `Runtime` is the composition root and owns the service graph.
- `runtime.agents.get(id)` returns a Runtime-bound `Agent`; it does not compile or register new definitions.
- `AgentBinding` is created per execution and pins the exact durable semantics, including the output contract.
- `Session` is bound to `AgentSpec.id`; retry/recovery remain pinned to the exact historical execution binding.

## Shared run budgets

Use `RunBudget` at Agent, Session, or TaskGraph admission to share request/tool
counts, observed tokens, and an admission deadline across the execution tree.
See [shared run budgets](docs/run-budgets.md) for concurrency, unknown usage,
continuation, and durable recovery guarantees.

## Evaluation

Use `runtime.evaluations` to evaluate Tasks, Agent-backed Tasks, and native TaskGraphs with retained evidence, rule/model/human scorers, rescoring, and paired reports. See the [evaluation guide](docs/evaluation.md) for an offline end-to-end example and the capture, comparison, authorization, and retention contracts.

## 1. Run a workspace

### Command line

```bash
ai-run "review this change" --project /workspace/project --model gpt-4o-mini
python3 -m linktools ai run "review this change" --project /workspace/project --model gpt-4o-mini
```

Useful options:

- `--base-url`, `--api-key`, and `--model` also read `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.
- `--storage filesystem|sqlite` selects the Runtime storage backend.
- `--planning` enables planning for the execution.
- `--thinking` requests model thinking when supported.
- `--json` emits one terminal JSON result.

### Python

```python
from linktools.ai import CapabilityGroup, Runtime, Workspace
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import RuntimeStorage

workspace = Workspace.initialize("/workspace/project")
models = ModelRegistry.openai(model="gpt-4o-mini")
storage = RuntimeStorage.in_memory()

async with Runtime.open(
    "default",
    models=models,
    storage=storage,
    capabilities=(CapabilityGroup("workspace", workspace=workspace),),
) as runtime:
    result = (await runtime.agents.get("default").run(
        "review this change",
        memory_scope="default",
        planning=True,
    )).result
```

`Runtime.open()` is the public composition root. The Runtime composition is immutable for the lifetime of the context; registrations are completed before it opens.

Connection settings can be resolved lazily for a route. The resolver runs only
when that route is materialized, and each materialization resolves independently;
there is no process-global first-use client cache or initialization lock. Concurrent
first use may therefore create independent provider instances. An alias keeps the
concrete binding that was registered as its target:

```python
models.register_openai(
    "production",
    model="gpt-4o-mini",
    connection_resolver=lambda: {"api_key": get_api_key()},
)
models.register_alias("default", "production")
```

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
    "default",
    models=models,
    storage=storage,
    capabilities=(CapabilityGroup("workspace", workspace=workspace), application),
) as runtime:
    result = (await runtime.agents.get("audit").run("inspect ticket SEC-123")).result
```

Named behavior identity is exactly `(kind, id, revision)`. Agent, Tool, Skill, MCP, Capability, Task, and TaskExpander do not maintain a second hash/digest identity. Full declarations and execution-bound contracts are still persisted for exact restore and same-revision drift validation. `CapabilityGroup.tool()` and `CapabilityGroup.capability()` default to revision `1`; Agent/Skill/MCP declarations also carry revision `1` unless explicitly changed. Generic Pydantic capabilities retain their native Pydantic AI behavior, and LinkTools revalidates final output against the durable `OutputBinding`.

Generic capabilities are trusted host-Python extensions. LinkTools preserves their native hooks and does not sandbox or deny their file, network, or process access; only LinkTools-owned workspace, opaque-effect, and deferred-resolution boundaries provide those controls.

LinkTools-owned Tool declarations carry their runtime metadata in
`ToolDefinition.metadata`. `CapabilityGroup.tool()` writes the registered
`effect_policy`, `plan_safe`, and `tool_class=business` values there; Workspace
and MCP owners declare their own effect policy, class, path, and context
metadata. Plan filtering, sandbox selection, leaf effect handling, and compaction consume
these declarations instead of inferring behavior from Tool names.

`CapabilityGroup.agent()` creates an `AgentSpec`; the declaration carries its explicit positive `revision` in the single v1 wire contract.

## 3. Workspace and declaration assets

`CapabilityGroup("workspace", workspace=workspace)` contributes the stable
Workspace tools and sandbox boundary only. It does not discover Agent, Skill, or
MCP declarations from Workspace paths.

Agent, Skill, MCP, and repository rule files use one explicit source: pass a ready
`AssetStore` with `CapabilityGroup(..., assets=store)`. A store-backed group
captures the declaration metadata visible when capture begins. Assets added
afterward are ignored for that capture; assets actually read by a loader must
still match their captured metadata through verification. Conflicting
identities or layouts fail closed.

Every Asset kind uses `AssetVersionRef` for byte-version reads through
`AssetStore.resolve_versions()` and `read_versions()`. Agent, Skill, MCP, and
Rule declarations are read from the versions captured by the group. Writable
Asset backends retain historical versions; `DirectoryAssetBackend` is a
read-only view of local files and ignores the requested revision by default.
For that backend, `AssetStore` verifies the current bytes against the captured
size and digest, so changed or missing content cannot satisfy a version read.
Declaration format versions and named revisions are separate from Asset
byte versions.

Repository rules use the `rule` Asset kind and Markdown keys such as
`AssetKey("rule", "review.md")` or `AssetKey("rule", "python/strict.md")`.
Rule Markdown is an instruction document at root scope; its content is not
reinterpreted as Workspace configuration. Captured Rule instructions work with
or without a Workspace. When a Workspace exists, `AGENTS.md` is additionally
resolved from the Workspace directory. The Workspace's `.linktools/rules`
directory is not an implicit rule source. A directory-backed AssetStore can
expose local rule files through `DirectoryAssetBackend` and a
`PrefixAssetPathAdapter`.

For downstream declaration formats or custom kinds such as `worker` or
`audit`, implement `CapabilityLoader` and register it for its input Asset
kind with `group.loader("audit", loader)`. One kind has exactly one loader;
registering `agent`, `skill`, `mcp`, or `rule` replaces that built-in
parser slot instead of chaining with it. Every loader receives the same
`CapabilityLoadContext`: `read()` / `read_many()` read the captured
immutable versions, while `bind_versions()` returns their `AssetVersionRef`
values for execution-bound resources. Loaders return normal
`CapabilityContribution` values, or `RepositoryInstructionDocument` for
instruction-only inputs such as Rule. Use
`CapabilityContribution.from_declaration(...)` for Agent, Skill, and MCP
declarations. Resource-backed declarations remain owned by the same
CapabilityGroup AssetStore; a custom loader may change declaration format or
layout but does not implement a second version store. No additional
Registry/Provider abstraction is required.

The built-in file layouts are deliberately small: Agents use
`<id>/AGENT.md`, Skills use `<id>/SKILL.md` or `<id>/skill.md`, and MCP servers
use `<id>/mcp.json` or `<id>/mcp.yaml` for every supported transport. Flat Agent/Skill/MCP declaration
files are not a second built-in authoring path. Authoring adapters validate the fields they consume without rejecting unrelated
ordinary fields. `AgentSpecAdapter`, `SkillSpecAdapter`, and
`MCPServerSpecAdapter` convert supported authoring inputs into their matching
Spec values, while the corresponding `*SpecCodec` classes own durable
serialization and contract projections. Custom source kinds such as `worker`
can use `AgentDeclarationLoader("worker", defaults=...)`; explicit Agent
fields still override validated defaults.

Captured resource paths must also be portable: use relative POSIX paths with
filenames valid on Windows as well as Unix. Colons, Windows reserved names,
and trailing dots or spaces are rejected consistently during capture and
materialization, including for in-memory assets.

Skill filenames accept the two spellings supported by the Agent Skills
[reference parser](https://github.com/agentskills/agentskills/blob/69ef37e9424c0a7ea9dd2293b559e43ec8176379/skills-ref/src/skills_ref/parser.py).
Use `SKILL.md` for portability: other clients may require that exact spelling.
Both files in one package are rejected as conflicting declarations, just like
multiple MCP declarations; LinkTools does not silently prefer one. Other
declaration filenames, package IDs, resource paths, and field names retain
their existing case-sensitive matching.

`AGENT.md` and `SKILL.md` may include a `metadata` map. Skill metadata is
not shown to the model. The optional `metadata.linktools-revision` value may
be a positive integer or its decimal string representation and maps only to
the named Skill revision. Other Agent Skills fields, including
`allowed-tools`, remain author content and do not grant Runtime permissions.

Shared MCP configuration uses the common `mcpServers` JSON shape through
`MCPServerSpecAdapter.decode_config(data, revision=...)`. LinkTools does not
auto-discover a project `.mcp.json` or treat arbitrary flat MCP JSON Assets as
shared configuration. The host or a custom `CapabilityLoader` explicitly
chooses the source and revision, then registers each resulting
`MCPServerSpec` with `CapabilityGroup.mcp(server)` (or returns it from a
loader) into the same `CapabilityGroup.capture()` path.
Supported transports are stdio, Streamable HTTP (`http` and
`streamable-http` authoring names), and explicit legacy `sse`; LinkTools
does not fall back from one remote transport to another.

`CapabilityGroup.capture()` returns a `CapabilityGroupCapture`. Pass that
capture to `Runtime.open()` when the host also needs to inspect the same
captured contributions; this avoids parsing the source declarations twice.
The capture contains the group id, contributions, captured Rule instructions,
source revision, and Workspace association. A changed source revision fails
Runtime admission. Runtime reads captured resources through `AssetStoreReader`,
a read-only view that does not expose the mutable `AssetStore`.

Directory-backed Skill packages retain a native absolute package path when
all effective Skill assets under the declared resource root map to one
consistent local package tree. The declaration filename is loader-defined; it
does not need to be `SKILL.md`. This allows Skill scripts to be invoked by
absolute path without letting a single file symlink redefine the package root.
If an overlay mixes resource origins, the declaration view is virtual rather
than claiming a single native package directory. Bubblewrap can still bind
its local files individually; LocalSandbox requires one matching native tree.
Durable executions pin Asset version
references for Skill resources and read them through AssetStore when needed.
The Sandbox exposes existing local resource files by path after verifying their
bound Asset versions and executable bits. Resources without native file paths
remain available through `load_skill` but cannot be executed by file path.
Skill resource bytes are not copied to a temporary directory or Runtime ObjectStore.
Because the paths point to original files, external edits after verification
can be observed by an already running process.

For a stdio MCP package, the package directory supplies the default resource
root when `resource` is omitted; an explicit resource root takes precedence.
The resource root is also that server process's working directory, so a plain
argument such as `server.py` works together with sibling imports and relative
data-file reads. Native directory resources retain their verified local paths.
For Asset backends without native file paths, Runtime materializes the captured
Asset versions into a private temporary package tree for each run. The process
closes before that tree is removed on success, failure, or cancellation; runs
do not share a persistent resource cache. Materialization preserves the
configured host-stdio or Sandbox execution boundary and does not copy resource
bytes into Runtime ObjectStore. If process cleanup cannot establish safe
closure, resources remain available rather than being deleted under a live
process.

The existing `resource:<relative>` argument form is also supported. Capture
rejects absolute paths, traversal, and missing files in these references and
binds package files to Asset version references. A Streamable HTTP or SSE MCP
package has no resource root, and neighboring files are neither projected nor
uploaded. Outside a resource-backed stdio package, a string such as
`resource:literal` remains an ordinary argument.

MCP connection values are live declaration data rather than durable semantic
identity. The execution pin keeps the server id/revision, canonical transport,
semantic args, resource refs and execution policy, but excludes
`command`, remote `url`, `env`, `headers`, `init_timeout` and `read_timeout`. Recovery therefore requires
the host to supply the current declaration with the same MCP id/revision;
connection values may change under that identity, while transport, semantic
args or resource semantics require a revision change.

### MCP files, Python registration, and shared configuration

All three entry points produce the same `MCPServerSpec` and use the same
selection, effect tracking, metrics, and recovery boundaries. Registration and
capture do not connect to servers; only servers selected by an Agent connect.
Every loaded declaration must still be valid, including its environment
references, even if an Agent does not select it.

The CLI reads its explicitly configured AssetStore. Its standard MCP location is
`workspace.storage_root / "mcp" / "<id>" / "mcp.yaml"` (or `mcp.json`);
custom stores choose physical paths through their path adapter. It does not
scan home directories, other workspaces, or `.mcp.json` implicitly.

Streamable HTTP needs only a URL. Use environment references for credentials:

```yaml
# mcp/security/mcp.yaml
url: https://mcp.example.internal/mcp
headers:
  Authorization: "Bearer ${MCP_TOKEN}"
init_timeout: 10
read_timeout: 60
```

Legacy HTTP+SSE must be explicit; a URL ending in `/sse` does not select it:

```yaml
# mcp/legacy/mcp.yaml
transport: sse
url: https://legacy.example.internal/events
headers:
  Authorization: "Bearer ${LEGACY_MCP_TOKEN}"
```

A stdio package can use a plain filename with any supported AssetStore:

```yaml
# mcp/local/mcp.yaml; server.py and its dependencies are in the same package
command: python
args:
  - server.py
env:
  API_KEY: "${LOCAL_MCP_API_KEY}"
```

No explicit `resource` field, temporary-directory option, or `resource:` prefix
is needed for this package layout. `args: [resource:server.py]` remains valid.
Direct Python registration still needs an explicit Asset resource root when
its files belong to an AssetStore rather than the host working directory.

JSON/YAML with only `command` defaults to `stdio`; with only `url` it defaults
to `streamable-http`. `type: http` and `transport: http` are authoring aliases
for `streamable-http`. Supplying both `type` and `transport`, an unknown
transport, or incompatible connection fields fails validation. Remote
servers cannot carry stdio `command`, `args`, `env`, or `resource` values.
An explicit declaration ID must match its package or shared-config entry ID.

Direct registration does not need an AssetStore unless the declaration has
local resources. Python preserves the existing `transport="stdio"` default,
so specify the transport for remote servers:

```python
import os
from linktools.ai import CapabilityGroup
from linktools.ai.spec import MCPServerSpec, mcp_server_selector

group = CapabilityGroup("services")
server = group.mcp(MCPServerSpec(
    "security",
    transport="streamable-http",
    url="https://mcp.example.internal/mcp",
    headers={"Authorization": f"Bearer {os.environ['MCP_TOKEN']}"},
    init_timeout=10,
    read_timeout=60,
))
assert server.id == "security"
group.agent("reviewer", allow_tools=(mcp_server_selector("security"),))
# Pass capabilities=(group,) to Runtime.open(...).
```

Import a shared configuration explicitly, then register its declarations:

```python
from pathlib import Path
from linktools.ai import CapabilityGroup
from linktools.ai.spec import MCPServerSpecAdapter

group = CapabilityGroup("services")
servers = MCPServerSpecAdapter().decode_config(
    Path("services.json").read_bytes(), revision=1,
)
for server in servers:
    group.mcp(server)
```

```json
{
  "mcpServers": {
    "security": {
      "url": "https://mcp.example.internal/mcp",
      "headers": {"Authorization": "Bearer ${MCP_TOKEN}"}
    }
  }
}
```

Environment expansion belongs only to the authoring adapter and only to
`url`, header values, and environment values. `${NAME}` expands once; `$$`
escapes a literal dollar sign. Expansion is not recursive, ordinary `$NAME`
is literal, and shell/default-value expressions and `.env` discovery are not
supported. Missing variables or malformed references fail with
`OUTPUT_CONTRACT_INVALID` without exposing their values. IDs, revisions,
commands, arguments, keys, and resource paths are never expanded. Python
Specs and durable wire decoding preserve strings literally.

Capture freezes the resolved connection configuration. Rebuild the capture
and Runtime to rotate credentials; calls do not reread the environment or
configuration. Static headers are service credentials, not credentials
inherited from a Principal or incoming request. Wire declarations contain the
connection configuration, but Runtime binding/history does not copy resolved
credentials into durable MCP contracts. Source files and backups containing
literal secrets remain sensitive.

`init_timeout` and `read_timeout` are optional keyword-only connection values
in seconds. Each must be finite and positive; booleans are invalid. Omitted
or `None` values preserve FastMCP Client defaults. An explicit `init_timeout`
bounds the complete transport startup, including SSE endpoint negotiation;
`None` keeps the SDK's stage-specific defaults. `read_timeout` governs
upstream request waiting, not the total Agent or Task deadline. Legacy SSE
also applies an explicit value to its SSE read timeout. HTTPS keeps SDK
certificate verification. There is no automatic anonymous downgrade,
OAuth login/refresh, transport fallback, or custom TLS/proxy interface.

Each selected MCP gets an independent Client per run. Connections are not
shared across concurrent runs, Subagents, tenants, or Runtimes. The SDK owns
protocol negotiation and supported tool-list cache invalidation; LinkTools
keeps its own tool names, selection, and `non_replay_safe` effect policy.
A server's replay-safety hints do not grant permission to replay writes.

Expected connection, negotiation, and discovery failures become
`MCP_CONNECTION_FAILED`; final Client destruction failures become
`MCP_CLEANUP_FAILED`. Safe diagnostics identify the server, transport, phase,
and available HTTP status, without recording URLs, headers, response bodies,
or credentials. Existing typed sandbox errors and cancellation retain their
meaning. Cleanup attempts every owned connection and does not replace a
primary failure with a cleanup failure.

A disconnect, timeout, or cancellation after a tool begins can leave its
external result unknown. Existing ToolOperation recovery requires confirming
that result before resuming; closing a connection is not a remote rollback.
A durable confirmed result is reused even when later cleanup fails. Recovery
creates a new Client from the currently supplied same-ID/revision declaration;
network session IDs, SSE cursors, and Client objects are not persisted.

Changing an address or credential under the same ID/revision must still reach
the same logical service, data domain, and behavior. Changing tenant, scope,
tool behavior, transport, or other semantic meaning requires a new ID or
revision. Asset capture does not freeze a remote server's implementation.
Resources/prompts APIs, OAuth, dynamic end-user credentials, mTLS, server
publishing, and arbitrary third-party configuration import are out of scope.

### Execution sandbox

Workspace filesystem and shell tool effects run through the public `Sandbox` / `SandboxSession` boundary. Inject a custom implementation with `CapabilityGroup(..., sandbox=...)`; the Workspace can come from the same or another group. A Sandbox can also be configured without a Workspace for Skill resource paths and MCP stdio.

When no Sandbox is configured, LinkTools uses its built-in `LocalSandbox`
adapter for supervised host processes, including stdio MCP without a Workspace.
This does not provide OS isolation. LinkTools owns the stable model-visible workspace
tool signatures, descriptions, metadata, and durable capability pins.

A run opens a `SandboxSession` when it needs a selected Workspace filesystem/shell tool, a local Skill resource path, or a stdio MCP server assigned to that Sandbox. Workspace tool commands share that session, which is closed when the model run succeeds, fails, or is cancelled. Resource-backed stdio uses its package directory as its process working directory; other stdio uses the Workspace execution root, or the host current directory captured when Runtime opens if no Workspace is present. Streamable HTTP and SSE MCP connections use the `host-network` boundary,
regardless of the configured Sandbox. A remote-only Agent with no selected
local capabilities or Skill resources does not open a Sandbox session and
needs no Workspace or working directory. In a mixed run, stdio remains in
its configured `host-stdio` or `workspace-stdio` boundary while HTTP remains
in the Runtime process network. Here "host" means the environment where
Runtime actually runs: in a container, `localhost` refers to that container.
A Bubblewrap network restriction does not revoke an Agent's explicitly
selected remote capabilities. Use capability sources and `allow_tools` to
withhold them. Remote tools receive explicit arguments, not automatic access
to Workspace files, roots, mounts, or host resources.

Use `DisabledSandbox` to keep workspace tool declarations and historical binding recovery available while making runtime workspace tool materialization fail with `SANDBOX_UNAVAILABLE`. A custom Sandbox failure does not fall back to the local host environment.

`LocalSandbox` runs Workspace commands with the selected execution root as its current directory; a resource-backed MCP process uses its own package root.
It is an execution boundary, not an operating-system security boundary. On Linux,
`BubblewrapSandbox` is an explicit deployment choice. It requires a non-root
user, usable unprivileged namespaces, `bwrap >= 0.12.0`, and a trusted
read-only runtime rootfs containing the same LinkTools build and Python >=
3.10. It has no automatic Local fallback. Bubblewrap isolates Session
file/command execution and sandboxed MCP stdio. The host Agent, model
requests, and Python custom tools remain outside it. A restricted MCP process
sees the selected execution root under its configured read policy and its own
declared resources as read-only, with no external network route. LocalSandbox provides supervised trusted host stdio and is the Runtime default
when no Sandbox is explicitly selected; it does not claim OS isolation.
Read-policy-restricted LocalSandbox sessions reject stdio rather than bypassing
their policy. DisabledSandbox and file-only custom sessions still reject MCP
stdio.

Selected local Skills are exposed at `/resources/r<hash>` in Bubblewrap through
read-only file mounts and at their original package location in Local sessions.
The mapping is derived for the current run and is not persisted into Skill
declarations. Background command state is ephemeral and is cleaned up
with the Session; it is not a cross-run service.

For a read-only Runtime session, configure the supported backend with one
immutable policy. Rules are root-relative POSIX patterns; an empty rule set
denies reads, and resource rules are keyed by `SandboxResource.id`:

```python
from linktools.ai import CapabilityGroup, Workspace
from linktools.ai.workspace import LocalSandbox, ReadOnlySandboxPolicy

readonly = ReadOnlySandboxPolicy(
    readable_paths=("src/**", "README.md"),
    resource_paths={"review": ("**",)},
)
workspace = Workspace.load("/workspace/project")
group = CapabilityGroup(
    "workspace",
    workspace=workspace,
    sandbox=LocalSandbox(read_policy=readonly),
)
```

With a read-only policy, Bubblewrap mounts the Workspace read-only and never creates hidden-path mount points there. Provision its hidden directories (including `.linktools`) before opening the sandbox; a missing or unsafe mount point causes startup to fail with `SANDBOX_UNAVAILABLE`.

## 4. Agent selection and capability policy

`AgentSpec` contains declarative selection policy:

```python
from linktools.ai.spec import AgentSpec

spec = AgentSpec(
    id="audit",
    model="default",
    system_prompt="Audit the supplied change.",
    instructions=("Cite concrete evidence.",),
    allow_tools=("read_file", "mcp:security:*"),
    allow_skills=("review",),
    allow_subagents=("triage",),
    planning=True,
    thinking="high",
)
```

The compiler resolves these selectors once from the captured candidate universe.
Missing or conflicting required candidates fail closed. MCP selectors use
`mcp:<encoded-server>:<encoded-tool>` or `mcp:<encoded-server>:*`; use
`mcp_server_selector()` and `mcp_tool_selector()` to encode logical ids and
upstream tool names. For example, `security/audit` and `scan:file` become
`mcp:security%2Faudit:scan%3Afile`. The `mcp__` prefix is reserved for
model-visible transport names and is not an authoring selector.

Model-visible control text is partitioned by lifecycle. Binding-static Agent instructions, Skill catalog/preloads, and Workspace guidance form the most stable prefix. Execution-static capability guidance and the repository instructions already active when the Execution starts follow that prefix and remain fixed for that Execution. Repository sources first discovered later through declared workspace path fields are kept in one refreshable overlay; only that overlay changes when a new scope becomes effective. Provider adapters may serialize these instruction parts as `system`, `developer`, or another supported instruction channel, and the physical request may still carry the same fixed prefix on every model call so provider prompt caching can reuse it.

The fixed prefix is not appended as a new conversation message on each model call. Lazy Skill bodies/resources, Memory contents, the current-plan reminder, user input, tool and Subagent results, attachments, and retry/error feedback remain dynamic request or conversation context. Memory guidance is fixed only for an Execution that enables Memory; Memory contents stay pull-based through the Memory tools, so updates from another Execution are observed when the Agent reads/searches Memory again rather than being pushed into the standing prompt. Workspace instruction text describes stable usage constraints only; authorization and approval remain Runtime-enforced behavior. Recorded requests may therefore contain the same fixed instruction parts repeatedly without representing repeated historical messages.

`allow_tools` controls ordinary/external model-visible tools. `mode="plan"` selects plan mode for an execution; `planning` controls whether the Agent has the planning capability. Plan mode implies `planning=True`. Runtime infrastructure capabilities such as planning, memory, Skill loading, and Subagent delegation are composed by Runtime according to the resolved execution contract.

Subagents are root Agent definitions selected from the same captured catalog. A root Agent cannot select itself as a Subagent, and the Runtime does not create a second registration system for child Agents.

## 5. Output contracts

Output belongs to an execution, not to `AgentSpec` or Session identity:

```python
from pydantic import BaseModel

class Finding(BaseModel):
    title: str
    severity: str

agent = runtime.agents.get("audit")
result = (await agent.run(
    "inspect the patch",
    output=Finding,
)).result
```

The exact durable binding stores:

- the v1 `AgentSpec` contract;
- the resolved model contract;
- the selected capability pins, including bound Skill Asset version references when Execution state is durable;
- selected Subagent ids and their direct execution bindings;
- `output_mode`;
- the canonical output JSON Schema;
- one `binding_digest`.

The binding contract does not persist Python output import paths or duplicate identity hashes. `ExecutionResult` exposes the derived `output_contract_digest` together with the terminal output.

## 6. Sessions and executions

```python
agent = runtime.agents.get("audit")
session = await agent.create_session("chat-1")

first = (await session.run("inspect the first change")).result
second = (await session.run(
    "return a structured summary",
    output=Finding,
    planning=True,
)).result

history = await session.history()
```

A Session owns conversation continuity and the stable Agent id. Every new execution binds the current compiled Agent to that execution's output contract. Retry, fork, durable recovery, evaluation, and Task execution use the exact binding contract/digest required by their contract rather than re-running current selector discovery.

User prompt transport is also durable: plain text uses the `text` codec, while supported native Pydantic user content uses the v1 durable user-content codec. Unsupported external file lifecycle objects fail closed instead of being guessed or silently converted.

URL and uploaded-file content remains an external reference: Runtime persists
its declared metadata and does not implicitly download it. Inline binary
content and workspace file inputs are materialized as bytes before execution
reservation when their durable contract requires it. Model transport retries
use `max_retries=2` with a fixed `retry_delay=1.0`; tool correction defaults
to `tool_retries=10`, and output correction defaults to `output_retries=3`.

OpenAI model declarations accept explicit `vision=True` or `vision=False`
(`False` by default). The value is part of the model model digest. A
model with `vision=False` rejects recognizable image content at the final
request boundary before provider calls or transport retries; PDF, text, and
other non-image attachments continue through the existing provider contract.
LinkTools does not infer image support from model names, endpoints, or probes.

Execution file input uses the same durable boundary:

```python
result = (await agent.run(
    "分析这些截图",
    files=("screenshots/overview.png", "screenshots/details.png"),
)).result
```

When a file must be captured as part of the accepted prompt, use the pure
`WorkspaceFileInput` value. Runtime reads it through the configured Sandbox and
preserves its order and optional opaque identifier:

```python
from linktools.ai.core import WorkspaceFileInput

result = (await agent.run(
    ("Review this evidence:", WorkspaceFileInput(
        "evidence/report.txt",
        media_type="text/plain",
        identifier="report-1",
    )),
)).result
```

The Sandbox canonicalizes logical paths before reading them and preserves every input occurrence. Passing the same path twice therefore produces two attachment occurrences with distinct execution-local `attachment_id` values, while their content digests may be identical. The initial model request receives each file as `BinaryContent` together with its canonical Workspace path, and the captured bytes are recovered from Runtime storage rather than reread from the Workspace during retry or recovery. After a complete model response consumes that binary input, Runtime keeps only lightweight file/path context in the active model context, so later agent-loop requests, Session turns, and forks do not repeatedly resend the bytes. The raw transcript remains lossless.

If an Agent needs to inspect a Workspace file again, select `attach_files` in `allow_tools`. `attach_files(paths=[...])` is a normal `filesystem.read` Workspace tool: it applies the existing Sandbox, path, approval, and repository-instruction boundaries, preserves duplicate occurrences, reads the current Workspace contents, and sends those files only to the next model request. Runtime binds those occurrences to the originating tool call before the content enters model history, so parallel tool calls do not require transcript-order inference. A later complete model response consumes them under the same transient rule. Agent-backed Task nodes use `runtime.tasks.from_agent(...)` and `AgentTaskInput`; place workspace paths in its `files` field and prompt content in `prompt`. Runtime materializes those inputs at graph admission, or when accepting a dynamically expanded batch, before dependent nodes run. Replaying an accepted graph does not reread the source files; graph state retains the captured input objects. Delegated subagents use the same explicit execution-input rules.

Attachment delivery evidence is available through `runtime.history.attachment_facts(...)` and `RuntimeHistory.open(...)`. The structured facts distinguish `accepted` from `included_in_request`, expose known media type/size/digest, request association, and the optional opaque `input_identifier` originally supplied by the caller. Runtime does not interpret or synthesize that identifier for Workspace or `attach_files` inputs, and keeps `processing_status="unknown"` unless it has verifiable provider-specific evidence. External URL references are not downloaded just to manufacture size or digest facts.

### Agent-backed Task graphs

Bind Agent-backed Task definitions through `runtime.tasks` and put prompt,
parameters, and Workspace paths in `AgentTaskInput`:

```python
from linktools.ai.runtime import AgentTaskInput
from linktools.ai.task import TaskGraph, TaskNode

task = runtime.tasks.from_agent("writer-task", runtime.agents.get("writer"))
engine = runtime.tasks.bind(task)
graph = TaskGraph(
    "draft-1",
    (TaskNode("draft", task=task, input=AgentTaskInput("Say hello.")),),
)
run = await engine.start(graph, idempotency_key="draft-1")
await run.wait()
result = await run.result("draft")
```

Agent task prompt content supports native user content such as `BinaryContent`
and `WorkspaceFileInput`. Runtime stores its captured bytes as part of the
accepted graph input so retries and recovery do not reread the source files.

### Runtime context and execution queries

`Runtime.open()` is the stable Runtime construction entry point. Use
`RuntimeContext` to provide the application object, tenant, correlation, and
low-cardinality metric dimensions for the Runtime lifetime:

```python
from linktools.ai.runtime import RuntimeContext

context = RuntimeContext(app, tenant_id="tenant-a")
async with Runtime.open(
    "web-chat",
    models=models,
    storage=storage,
    context=context,
) as runtime:
    ...
```

Execution metadata is available through the authorized public query surface:

```python
from linktools.ai.runtime import ListExecutionRequest

page = await runtime.executions.list(
    ListExecutionRequest(
        principal=runtime.default_principal,
        session_id="chat-1",
        agent_id="audit",
    )
)
```

`RuntimeHistory.open()` provides the same read-only execution metadata and
history projections without loading a ModelRegistry or compiling Agents.
`inspect_execution()` returns safe durable summaries for binding, input,
output, timestamps, usage, and errors; it does not return prompt/output bodies
or raw error diagnostics. `result()`, `history()/trace()`,
`transcript()/model_interactions()`, `task_graph()`,
`list_execution_events()/list_task_events()`,
`usage()/graph_usage()`, `attachment_facts()`,
`task_result()/task_result_ref()`, and `artifacts()` are owned by the same
authorized query composition. Execution list and detail cursors are opaque namespace-bound continuations;
authorization is rechecked for every query. Paged detail queries fix the
committed high-water captured by the first page; restarting from the first page
can observe newer facts.

Execution and TaskGraph watch events carry opaque resumable cursors. The cursor
is bound to namespace, tenant, resource identity, content mode, and the durable
positions of the relevant execution streams. A cursor produced with
`include_content=False` cannot be reused with `include_content=True`.
TaskGraph replay captures a fixed durable prefix and emits the same event model
with resumable cursors; it does not invoke models, task handlers, or external
systems. `dependency_policy="all_succeeded"` is the default and blocks a node
when a dependency cannot succeed. `all_terminal` waits for every dependency to
finish, then permits execution regardless of their outcomes. `any_succeeded`
also waits for every dependency to finish, then permits execution only when at
least one succeeded. With no dependencies, `all_succeeded` and `all_terminal`
are ready; `any_succeeded` is blocked. A dependency block is recorded as
`BLOCKED` with `TASK_DEPENDENCY_FAILED` and never invokes the handler.

Handlers that run after terminal dependencies receive `dependency_states` for
all declared dependencies, while `dependencies` contains result references
only for successful dependencies. `failure_policy="propagate"` is the default
and includes node failures and dependency blocks in the graph's final status.
`failure_policy="isolate"` keeps those node outcomes visible but excludes them
from the graph's failed/blocked aggregate. It does not change scheduling,
cancellation, recovery, or retry behavior. An isolated graph can therefore
finish `SUCCEEDED` while individual nodes failed; inspect node outcomes when
all-success matters. Cancellation remains part of the
normal node aggregate and is never isolated by `failure_policy`.

Whole-graph cancellation is a separate durable control intent. If cancellation
wins before the business graph reaches a terminal state, the graph converges to
`CANCELLED` after active work settles while preserving each node's actual
terminal outcome; `RECOVERY_REQUIRED` still takes precedence. Task cancel
callbacks and `TaskNodeRunner.cancel()` are replay-safe control cleanup hooks:
a `RUNNING` or `EFFECT_UNKNOWN` cancel may invoke them again after process
loss. Observer callback failures are reported as `OBSERVER_FAILED` and do
not fail the graph or retry nodes.

Execution and TaskGraph observation streams also expose
`MODEL_REQUEST_STARTED` and `MODEL_REQUEST_FINISHED`. The request key is
`(execution_id, agent_run_seq, model_request_seq)`; a started event means
the Runtime accepted a logical handler call, not that a provider received
network traffic. Default lightweight observation keeps only request identity,
purpose, retry index, status, timestamps, duration, safe error code, and
request-level usage. It never includes prompts or response text. A successful
handler call that later fails output validation remains successful, and the
retry is a separate request.

An uncursored `model_interactions(include_content=False)` query captures a fixed
set of request identities: it includes RUNNING requests and terminal requests visible when
the query starts. RUNNING items have known `started_at`; `finished_at`,
`duration_ns`, `usage`, and `error_code` are `None`. Usage is never shown as
zero before it is known. Passing explicit `cutoffs` fixes request identities to
those high-water marks; it does not switch to a second query mode. `cutoffs=()`
selects an empty snapshot. Cursors retain their captured high-water marks but
do not freeze lifecycle state: a captured RUNNING request may be terminal when
read later. Root queries include recursive descendants; selecting a SUBAGENT
reads only that execution. Model interactions can read process-local staging,
while aggregate `usage()` reads archived usage and may lag. See the
[history guide](docs/runtime-history.md) for query scopes and content budgets.

The live event buffer can fall back to durable replay before an uncommitted
start event is delivered. Clients that need to show every active request should
refresh the owning Runtime's `model_interactions(cursor=None, cutoffs=None,
include_content=False)` on subscription and reconnect, and while the execution
is active. Merge event and history rows by request key, with terminal state
taking priority over RUNNING. The history refresh is independent of the event
buffer and does not advance an event cursor.

Downstream code must not scan `ExecutionRecord`, codec data, `StateStore`, or
private repositories directly. The current pre-release wire contract stores
both policies explicitly on the single `task_node` wire type. Superseded
development shapes, including nodes without policy fields, are rejected rather
than guessed. Published-version fixtures and readers are added only when a real
compatibility commitment exists.

## 7. Runtime storage

`Runtime.open()` requires an explicit `RuntimeStorage`; storage selection belongs to the application:

```python
from pathlib import Path

from linktools.ai import Runtime
from linktools.ai.runtime import RuntimeStorage

runtime_root = Path("/var/lib/linktools/runtime")
storage = RuntimeStorage.sqlite(runtime_root / "runtime.db")

async with Runtime.open(
    "service-runtime",
    models=models,
    storage=storage,
) as runtime:
    ...
```

`RuntimeStorage` supports in-memory, filesystem, SQLite, and SQL backends. Repository domains keep their ownership, transaction, recovery, and retention rules; `Runtime.open()` consumes the storage composition directly. Offline export requires a caller-owned `SnapshotExclusiveGuard` that quiesces related writers and object cleanup; a read-only `RuntimeStorage` handle alone is not that boundary. The supported archive flow is: quiesce writers and object cleanup, export through a read-only `RuntimeStorage`, restore into an empty staging root, verify required history and object references, then let the application publish that staging root. `restore_snapshot()` restores data but is not itself an atomic publication primitive. RuntimeStorage snapshots include Runtime-owned objects only. Skill and MCP resource bytes remain Asset-owned and are persisted as Asset version references in
binding metadata; portable RuntimeStorage restore therefore requires the
corresponding Asset history to remain available through the AssetStore supplied
when the Runtime is reopened.

SQLite-backed `RuntimeStorage` supports the built-in durable TaskGraph scheduler without a SQLite-specific launcher or an external lock. Normal internal Task optimistic-CAS races are reread and converged by the Task domain. Durable ToolOperation terminal persistence is also lease-aware: a same-lease heartbeat racing terminal persistence is reconciled without replaying the tool effect. Genuine ownership, fence, idempotency, tool-result, effect-unknown, integrity, and storage errors remain observable. A newly created local SQLite database initializes its Runtime and `ai_objects` schema; an existing SQLite database is only validated and is never implicitly migrated or repaired. When `object_store` is omitted, durable SQLite Runtime objects are stored in the same database through the built-in `ai_objects` and `ai_object_chunks` tables. An explicitly supplied ObjectStore remains available when object payloads should live outside SQLite. External SQL backends still require explicit schema provisioning/migration. Process workers must initialize their own Runtime and SQL engine inside the worker process; initialized Runtime, engine, session, or connection objects must not be reused after `fork()`.

Durable local execution and recovery are provided by Runtime persistence and recovery
checkpoints and do not require an external workflow server. Harness provides the
Planning, Memory, StepPersistence, and context-compaction capability behavior,
while LinkTools remains the durable owner of plans, Memory records and mutation
receipts, execution history, and the raw transcript. Memory content continues to
persist through `MemoryRepositories` and `ObjectStore`. Compaction only rewrites the
request context projection; it never rewrites the raw transcript.

### Workspace relocation

Workspace has no independent persistent identity. `Workspace.root`, Runtime storage paths, SQLite paths, SQL endpoints, ObjectStore locations, and storage topology are deployment details. The Runtime persistence identity remains the explicit `namespace` plus tenant supplied to `Runtime.open()` / `RuntimeStorage`.

Moving a Workspace therefore does not require preserving or regenerating a Workspace ID. Restore Workspace files and Runtime storage consistently, reopen the Runtime with the same logical namespace and tenant, and normal durable recovery continues to use the captured execution inputs and
Asset-version-pinned Skill resources. A separate `Runtime.restore()` migration step is not required.

## 8. Execution failure diagnostics

`execution-error-diagnostics-v1` extends failed execution results with durable diagnostic context while keeping the existing safe error contract unchanged:

```python
result = (await execution.wait()).result

result.error_code
result.safe_error_details
result.error_diagnostics
```

`error_code` remains the stable machine-readable classification used by Runtime control flow. `safe_error_details` remains the redacted/safe mapping for ordinary application handling. A failed execution may additionally expose `error_diagnostics` with:

- `exception_type`: original exception class name, at most 256 Unicode code points;
- `exception_message`: `str(original_exception)`, at most 2048 Unicode code points;
- `cause_digest`: 64 lowercase hexadecimal SHA-256 characters computed from the untruncated original exception type and message.

`exception_message` is diagnostic data, not a safe/redacted field. It may contain sensitive text already present in the originating exception. Runtime does not add prompts, model responses, headers, request payloads, or URLs to diagnostics, but it also does not redact the exception message. Applications should apply the same authorization and retention controls to diagnostics as to other execution investigation data.

Diagnostics are durable with the failed execution and are returned after Runtime restart through the normal public execution result and terminal event APIs. Current failed records carry an explicit diagnostic field; malformed records missing it are rejected. Successful and cancelled executions never carry diagnostics.

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

Package-specific public contracts remain available from their owning packages,
for example `linktools.ai.asset`, `linktools.ai.model`, `linktools.ai.spec`,
`linktools.ai.capability`, `linktools.ai.workspace`, and `linktools.ai.runtime`.
These include `AgentSpecAdapter`, `SkillSpecAdapter`, `MCPServerSpecAdapter`, `AgentDeclarationLoader`,
`CapabilityGroupCapture`, MCP selector helpers, `WorkspaceToolDeclaration`, and the
optional stdio sandbox protocols. `ErrorDiagnostics` is available from
`linktools.ai.errors`.

Private modules prefixed with `_` are implementation details. Downstream applications should not import Runtime execution infrastructure, state repository internals, or private compiler helpers directly.

### Watch and wait for operations

Execution, TaskGraphRun and EvaluationRun expose `watch()` and
`wait(on_event=...)`. All SDK waits and Agent/Session run/plan return
`WaitResult` with `result`, the acknowledged `cursor`, and optional
`observation_error`. `include_event_content` controls callback payloads only;
`watch` retains its `include_content` option. Graph wait additionally uses
`include_content` to choose whether its authoritative result includes raw node
inputs. Node outputs are read with `result()` or `results()`.
See [operation observation and migration](docs/task-observation.md)
for recursive execution trees, evaluation graphs, cleanup guarantees, and the
breaking SDK and StepEvent durable-wire changes. Existing development data is
not automatically migrated or deleted.

`Agent.plan()` and `Session.plan()` start real planner executions and wait for
an observation boundary, selecting plan-safe tools. They are not dry-run
previews and can invoke models and authorized tools. `run(planning=True)` stays
in run mode and enables the planning capability; it does not select plan mode. Evaluation `completion="complete"` means planned work
has settled; it does not assert that every target succeeded or every score is
valid. Inspect trial outcomes, score statuses, and report gates separately.
