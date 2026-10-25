# linktools-ai

`linktools-ai` requires Python >= 3.11 and provides the Agent runtime layer for LinkTools. The public composition model is intentionally small:

```text
namespace
    + RuntimeStorage
    + one RuntimeFeatures or RuntimeFeaturesCapture
        -> optional, independent Workspace and Sandbox via RuntimeFeatures(...)
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
- `RuntimeFeatures` is the only public registration/discovery composition unit. It can provide a Workspace or Sandbox independently and captures direct registrations and, when store-backed, one declaration view pinned to Asset version references.
- `AgentSpec` is a runtime-independent Agent declaration.
- `AgentCompiler` is the sole Agent-level selector. It resolves model, tool, Skill, MCP, capability, and Subagent candidates from the captured Runtime candidate set.
- `Runtime` is the composition root and owns the service graph.
- `runtime.agents.get(id)` returns a Runtime-bound `Agent`; it does not compile or register new definitions.
- `AgentBinding` is created per execution and pins the exact durable semantics, including the output contract.
- `Session` is bound to `AgentSpec.id`; retry/recovery remain pinned to the exact historical execution binding.

## Application capability filtering

`RuntimeFeatures(..., filter=...)` accepts an optional synchronous
`Callable[[AgentSpec, CapabilityContribution], bool]`. The compiler invokes it on
captured `agent` (delegation target), `tool`, `skill`, `mcp` and generic `capability`
contributions before resolving existing Agent selectors. The root Agent itself is
not a delegation candidate. A child is compiled against its own Agent declaration.
The callback may inspect public declaration metadata; the SDK does not reserve
business metadata keys, prescribe a type/tag system or serialize the callback.

```python
def available(agent, contribution):
    if contribution.kind != "skill":
        return True
    required = contribution.contract.get("metadata", {}).get("applies_to")
    return required is None or bool(set(required) & set(agent.metadata.get("tags", [])))

group = RuntimeFeatures("application", filter=available)
async with Runtime.open(
    "application", storage=storage, features=group,
) as runtime:
    ...
```

`AgentCompiler.filter_capabilities(spec, capture.contributions,
capability_filter=available)` exposes the same candidate filtering for host
inspection without constructing a model or Runtime. It returns visible declarations,
not a fully resolved executable toolset; a management catalog need not materialize
Workspace tools. Keep the unfiltered capture for global asset administration.

A capture retains the current in-process filter policy without serializing it into
assets or execution bindings. If the policy depends on captured metadata, use
`dataclasses.replace(capture, filter=build_filter(capture))` and pass that capture
to `Runtime.open(features=...)`. This keeps the same source revision and
reader without loading the assets again. `Runtime.open` has no separate filter
parameter.

`RuntimeFeatures` also owns the current host authorization policy and prompt
limits: `RuntimeFeatures("application", authorization=policy, limits=PromptLimits(...))`.
Capture it once and reuse that capture for `Runtime.open(features=...)`,
`RuntimeHistory.open(features=...)`, or restored snapshot history. For restored
history, features supplies only the authorization policy; the restored generation
owns the storage routes and cache. The policy
instance remains in process; it is not serialized as an execution identity.
Omitting authorization uses the Runtime tenant policy. Prompt limits constrain
model context and are separate from TaskGraph and snapshot limits.

For a fixed MCP response fixture, register
`ToolResponsesCapability(ToolResponseFixture(ref, reader))` from
`linktools.ai.capability` before capture. The fixture
keeps its immutable Asset reference and borrowed reader. `Runtime.open` has no
separate authorization, prompt-limit, or tool-response arguments.

With no callback, existing selection behavior is unchanged. Wildcard selectors
expand the remaining candidates; explicit unavailable selections fail.
Callback exceptions propagate and non-boolean results are rejected. Use a pure,
deterministic callback over the captured declarations and stable application inputs.
This is applicability filtering, not an authorization or OS isolation boundary.
MCP filtering is per server; existing tool selectors still restrict its dynamic
tools. Generic capabilities are filtered as whole contributions. Repository rules,
Task definitions and DAG nodes are separate domains, not capability candidates.

The final compiled selection feeds instructions, toolsets, Skill loading and
Subagent delegation. Binding pins and Asset version references retain that exact
selection. Historical restoration does not execute the current callback; new
compilations do. An asset-only snapshot freezes declarations, not application
callback code. Reuse captured execution bindings when historical selection must
remain fixed. Ordinary metadata and binding digest projections are unchanged.

## Shared run budgets

Use `RunBudget` at Agent, Session, or TaskGraph admission to share request/tool
counts, observed tokens, and an admission deadline across the execution tree.
See [shared run budgets](docs/run-budgets.md) for concurrency, unknown usage,
continuation, and durable recovery guarantees.

## Runtime-wide capabilities

Register native Pydantic AI behavior through the same `RuntimeFeatures` used for
Agent declarations. The default `scope="runtime"` applies host policy to every
Runtime-owned Agent run,
independently of `allow_capabilities` and `RuntimeFeatures.filter`. Use explicit
`scope="agent"` for selectable behavior included in durable Agent bindings:

```python
group.capability(global_rate_limit, id="host.rate-limit")
async with Runtime.open(
    "application", storage=storage, features=group,
) as runtime:
    ...
```

A host can define its own `GlobalRateLimit` using native model/tool hooks, shared
state and request/usage metadata. See [Runtime-wide capabilities](docs/runtime-capabilities.md)
for a runnable RPM/TPM example, streaming/cancellation behavior, quota accounting
and provider retry boundaries.

Context retention, interrupted-run recovery and detailed trace are independent
Runtime-scoped opt-ins: `ExecutionCapability(retain_history=True)`,
`CheckpointCapability()`, and `TraceCapability()`. Durable storage alone enables
none of these optional guarantees. `Runtime.open(..., metrics=dataset)` independently enables automatic
Agent and Task observations, including evaluation work in that Runtime. The
shared dataset remains caller-owned. Task, Evaluation and Artifact remain
services and need no capability wrappers. Session metadata and display history
are enabled by the default `SessionCapability`; use `SessionCapability(enabled=False)`
to disable them explicitly. Conversation adds retained model context for future turns
and execution forks. Execution commit timing belongs to
`ExecutionCapability(commit=CommitPolicy.sync())`, using `CommitPolicy` from
`linktools.ai.core`. See [feature guarantees and commit
boundaries](docs/runtime-capabilities.md#optional-execution-persistence).

## Evaluation

See [Task graph authoring](docs/task-graphs.md) for typed JSON results, conditional branches, batch expansion, collection, and submission-time diagnostics.

Use `runtime.evaluations` to evaluate Tasks, Agent-backed Tasks, and native TaskGraphs with retained evidence, rule/model/human scorers, rescoring, and paired reports. See the [evaluation guide](docs/evaluation.md) for an offline end-to-end example and the capture, comparison, authorization, and retention contracts.

## 1. Run a workspace

### Command line

```bash
ai-run "review this change" --project /workspace/project --model gpt-4o-mini
python3 -m linktools ai run "review this change" --project /workspace/project --model gpt-4o-mini
```

Useful options:

- `--model-api` selects `openai` (default), `openai-responses`, or `anthropic`. This is the wire protocol, not a vendor/model-name guess. The same option works for `ai-run`, `ai-acp`, and `ai-web`.
- Persist the selection in `.linktools/config.yaml` as `model_api: anthropic`, alongside `model: your-model`. `MODEL_API` / the existing `LINKTOOLS_MODEL_API` configuration can also select the protocol; an explicit CLI option wins.
- `--base-url`, `--api-key`, and `--model` are one-shot overrides. With Anthropic Messages, the corresponding configuration/environment fields are `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_MODEL`; both OpenAI protocols use the existing `OPENAI_*` fields. The selected protocol's existing LinkTools config resolver still supports `LINKTOOLS_`-prefixed configuration and its non-secret cache. Anthropic never falls back to OpenAI credentials or endpoints.
- Model selection order is explicit nonempty CLI value, selected-protocol LinkTools configuration, workspace `model`, then the selected-protocol native fallback. Standard LinkTools configuration itself reads both native and `LINKTOOLS_`-prefixed environment variables. Connection settings use explicit CLI value, selected-protocol LinkTools configuration, then the native fallback. Empty connection values retain the native SDK fallback to the selected protocol's environment; they do not disable environment credentials. Credentials are secret fields and are not persisted by configuration resolution.
- Install `linktools-ai[anthropic]` for Anthropic Messages, or `linktools-ai[openai]` for either OpenAI API; add `web` for `ai-web`, for example `linktools-ai[anthropic,web]`. Missing provider extras produce an installation hint before a Runtime is opened.
- Endpoint paths retain native SDK semantics: OpenAI normally uses a base ending in `/v1`; Anthropic normally uses the server root and appends `/v1/messages` itself. Configure a gateway according to its supported API, not by copying an OpenAI endpoint.
- Arbitrary `model_settings` JSON is a Python binding option, not a CLI option. It accepts SDK common/provider settings and nested `extra_body` for endpoint request fields; thinking translation uses the SDK directly, independently of whether a response contains thinking parts.
- Generic limits use `--context-window`, `--max-tokens`, `--timeout`, `--max-retries`, `--retry-delay`, `--retry-max-delay`, and `--retry-backoff-factor`. Workspace keys use the same words with underscores; environment configuration uses `LINKTOOLS_MODEL_CONTEXT_WINDOW`, etc. Context/output limits are tokens; timeout/delays are seconds. Explicit CLI values override LinkTools configuration/environment, then workspace values, then library defaults. See the [model configuration guide](docs/model-protocols.md#generic-limits-and-configuration-names).
- `--planning` enables planning for the execution.
- `--thinking` requests model thinking when supported.
- `--json` emits one terminal JSON result.

### Model configuration sources

The CLI retains the standard `ConfigField` resolution path. Protocol-specific fields
are selected after resolving the API, so configuring Anthropic does not require
passing connection options on every invocation:

```bash
export LINKTOOLS_MODEL_API=anthropic
export LINKTOOLS_ANTHROPIC_MODEL=claude-sonnet-4-5
export LINKTOOLS_ANTHROPIC_BASE_URL=https://api.anthropic.com
# Supply LINKTOOLS_ANTHROPIC_API_KEY through your environment/secret manager.
ai-run "review this change"
```

For either OpenAI API, use `OPENAI_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`
with the same `LINKTOOLS_` prefix, and select `openai` or `openai-responses`.
Unprefixed native variables are also accepted by the standard config resolver. API selection is a
protocol choice, not a provider-name or model-name guess.

The standard project profile is `.linktools.json` in the process working directory;
the user profile is `~/.linktools/linktools.json`. For example:

```json
{
  "config": {
    "MODEL_API": "anthropic",
    "ANTHROPIC_MODEL": "claude-sonnet-4-5",
    "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
  }
}
```

These files are distinct from the workspace's `.linktools/config.yaml`.
`--project` selects the workspace; it does not relocate the process's standard
profile. The local profile overrides the user profile. Inside standard config,
profile `config` fields take precedence over profile `env` fields and process
environment variables, followed by runtime overrides and persisted config. Within
an environment layer, an unprefixed field takes precedence over its `LINKTOOLS_`
form when both are set.
The CLI uses that existing precedence unchanged; explicit CLI model/connection
values override it for one invocation. Workspace `model` and native environment
fallbacks apply only when higher-priority values are absent, as described above.

Both protocols use `secret=True` for their API-key field. Reading configuration
or passing a CLI override does not save keys to the configuration cache or print
keys in web status. A key deliberately placed in a profile remains plaintext in
that user-authored file; prefer environment injection and never commit it. Existing
non-secret configuration cache values remain readable and are not overwritten by
CLI arguments.

### Local Web console

```bash
python manage.py install --editable 'linktools-ai[web]'
ai-web --project /workspace/project --model your-model
```

Open `http://127.0.0.1:8765` for conversations, live execution, history,
model/tool details, metrics and Runtime status. The command surface is
`ai-run`, `ai-acp`, and `ai-web` (also `lt ai run`, `lt ai acp`, and `lt ai web`).
The former history, session, status, trace and metrics commands are consolidated
into the Web console; their old entry points are removed without aliases.
See the [Web console guide](docs/web-console.md) for command coverage, local
security, persisted history and explicit recovery of stopped executors.

### Python

```python
from linktools.ai import RuntimeFeatures, Runtime, Workspace
from linktools.ai.capability import WorkspaceCapability
from linktools.ai.model import create_model
from linktools.ai.runtime import RuntimeStorage

workspace = Workspace.initialize("/workspace/project")
binding = create_model(protocol="openai", model="gpt-4o-mini")
storage = RuntimeStorage.in_memory()
features = RuntimeFeatures("workspace")
features.model("default", binding)
features.capability(WorkspaceCapability(workspace))

async with Runtime.open(
    "default",
    storage=storage,
    features=features,
) as runtime:
    result = (await runtime.agents.get("default").run(
        "review this change",
        memory_scope="default",
        planning=True,
    )).result
```

`Runtime.open()` is the public composition root. The Runtime composition is immutable for the lifetime of the context; registrations are completed before it opens.

Cold `Runtime` and `RuntimeStorage` imports load the complete durable schema set.
Include import and `Runtime.open()` initialization in the caller or deployment's
startup budget, and report readiness only after the context opens successfully.
`Runtime.open()` imposes no fixed startup deadline. Reuse the opened Runtime for
its service lifetime; execution deadlines do not replace a startup budget.

A feature loader reads model configuration for each agent run. Its resolved
binding is reused within that run; the next run reads updated configuration.
To give multiple routes the same static binding, register that object under each name:

```python
binding = create_model(
    protocol="openai", model="gpt-4o-mini",
    api_key=get_api_key(),
)
features.model("production", binding)
features.model("default", binding)
```

### Protocols, native request fields and thinking

Install `linktools-ai[openai]` for Chat Completions and Responses, or
`linktools-ai[anthropic]` for Messages; combine extras when using both.
The base package does not install either provider SDK.

`create_model(protocol="anthropic", model="claude-sonnet-4-5")` returns a
model binding without a route name. Protocols are `openai` (Chat Completions),
`openai-responses`, and `anthropic`. Register bindings in the same feature
container as tools and capabilities:

```python
from linktools.ai.model import create_model
from linktools.ai.runtime import RuntimeFeatures

features = RuntimeFeatures()
features.model("assistant", create_model(protocol="anthropic", model="claude-sonnet-4-5"))
# Or load one fresh binding for each agent run. The function is synchronous.
features.model("deferred", lambda: create_model(protocol="openai", model="gpt-4o"))
```

Runtime captures the route mapping at opening without evaluating every loader.
A callable is evaluated once per agent run. That run uses the same binding for
contract capture, model validation, tool rounds and retries; the next run reloads
it. Concurrent runs do not share mutable loader results. Client resources are
created and released per run. See [model protocols](docs/model-protocols.md) for
custom bindings, request-body extensions, restore boundaries and trace semantics.

`create_model(..., model_settings={"extra_body": {"enable_thinking": True}})`
forwards native body fields when the endpoint supports them. Common and
provider-specific SDK settings go directly in `model_settings`; unknown keys are
not automatically body fields. A host can pass its JSON/YAML mapping without a
custom `ModelBinding`. Explicit settings override top-level request defaults; raw
body fields can override standard fields, including token limits. Keep credentials
in connection configuration, not contract-captured semantic values. Operational
`timeout`, `request_timeout`, `max_retries` and `extra_headers` reach the SDK but
are excluded from the semantic contract. Explicit SDK `max_retries` selects SDK
retry policy and disables outer retries, including their delay/backoff controls.

Agent/run `thinking=` supports `"max"`. The default `None` inherits settings/native
capabilities; an explicit host effort takes precedence over application Thinking.
`False` explicitly requests the least thinking available on always-on models.
SDK provider/body precedence remains unchanged. CLI omission inherits; use
`--thinking` or `--no-thinking` for an explicit choice.
The pre-release `profile=` and `model_thinking_control` APIs are removed; see the
[migration guide](docs/model-protocols.md#native-thinking-and-migration).

## 2. Define application capabilities and Agents

Use `RuntimeFeatures` for direct application registrations:

```python
from linktools.ai import AgentContext, RuntimeFeatures, Runtime
from linktools.ai.capability import WorkspaceCapability

application = RuntimeFeatures[None]("application")
application.model("default", binding)
application.capability(WorkspaceCapability(workspace))

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
    storage=storage,
    features=application,
) as runtime:
    result = (await runtime.agents.get("audit").run("inspect ticket SEC-123")).result
```

Named behavior identity is exactly `(kind, id, revision)`. Agent, Tool, Skill, MCP, Capability, Task, and TaskExpander do not maintain a second hash/digest identity. Full declarations and execution-bound contracts are still persisted for restoration and integrity checks. Tool, Skill, MCP and generic Capability identity checks use only their ID and revision within the kind namespace; configuration equality is the caller’s responsibility, not a second identity check. Agent/Task declaration validation and resource access, sandbox policy and output validation remain separate boundaries. `RuntimeFeatures.tool()` and `RuntimeFeatures.capability()` default to revision `1`; Agent/Skill/MCP declarations also carry revision `1` unless explicitly changed. Generic Pydantic capabilities retain their native Pydantic AI behavior, and LinkTools revalidates final output against the durable `OutputBinding`.

Generic capabilities are trusted host-Python extensions. LinkTools preserves their native hooks and does not sandbox or deny their file, network, or process access; only LinkTools-owned workspace, opaque-effect, and deferred-resolution boundaries provide those controls.

LinkTools-owned Tool declarations carry their runtime metadata in
`ToolDefinition.metadata`. `RuntimeFeatures.tool()` writes the registered
`effect_policy`, `plan_safe`, and `tool_class=business` values there; Workspace
and MCP owners declare their own effect policy, class, path, and context
metadata. Plan filtering, sandbox selection, leaf effect handling, and compaction consume
these declarations instead of inferring behavior from Tool names.

`RuntimeFeatures.agent()` creates an `AgentSpec`; the declaration carries its explicit positive `revision` in the single v1 wire contract.

### Native SDK composition and Runtime hosting

`SkillCapability`, `SubagentCapability`, and the capabilities returned by
`workspace_capabilities(..., session=...)` can be composed directly with a native
Pydantic Agent. Supply their public source registry, delegate, or already-opened
sandbox session, and install the public SDK feedback adapter once:

```python
from pydantic_ai import Agent
from linktools.ai.capability import (
    PydanticToolControlCapability, SkillCapability,
)

agent = Agent(
    model,
    capabilities=[PydanticToolControlCapability(), SkillCapability(skills, sources)],
)
result = await agent.run("Use an available skill.", deps=agent_context)
```

`agent_context` is an `AgentContext` populated by the caller with its application
dependencies and authorized identity. It is not a Runtime service locator or
proof of authorization. The adapter converts `ToolCallRetry` and `ToolCallFailed`
into the SDK's model-facing feedback, preserving its retry budget. Unrelated
errors, native SDK controls, and cancellation are not translated. Runtime already
installs this same adapter; Runtime users do not register it again. Its public
`build_model_retry()` and `build_tool_failed()` class methods also serve deferred
recovery, so normal execution and recovery share one conversion owner.

Native SDK composition does not create Runtime admission, approvals, effect
journals, receipts, checkpoints, or recovery. Native capabilities may also be
registered with `features.capability(native_capability, revision=1, scope="agent")`;
their SDK lifecycle remains native. Selecting a capability does not individually
filter its internal toolset through `allow_tools`. Tools requiring Runtime's
managed guarantees must use the existing tool, MCP, or Workspace declaration
paths rather than assuming an arbitrary native toolset is managed automatically.

Capability names do not imply one execution mode. `AssetCapability` is consumed
during declaration capture, not native execution hooks. Session, Execution,
Checkpoint, Trace, and ToolResponses capabilities configure Runtime services.
Unbound host declarations do not supply those services to a native Agent: pure
capture/configuration declarations are rejected rather than silently ignored;
Execution, Checkpoint, and Trace require their public service bindings.

Use each existing `CapabilityContribution` to materialize independent native
result/input behavior. Bare `ToolResultsCapability` and `ReferencedInputsCapability`
are callback contracts, not automatic SDK execution boundaries. The model is
supplied separately, never to a capability:

```python
from linktools.ai.capability import CapabilityContribution

projection = CapabilityContribution.from_opaque(
    "capability", "projection", projector, revision=1,
)
inputs = CapabilityContribution.from_opaque(
    "capability", "documents", authorized_inputs, revision=2,
)
agent = Agent(
    model=model_binding.materialize(),
    capabilities=[
        PydanticToolControlCapability(),
        projection.materialize_capability(),
        inputs.materialize_capability(),
        SkillCapability(skills, sources),
    ],
)
```

`materialize_capability()` accepts Agent-scoped capability declarations, not
Tool/Agent/Skill/MCP authoring declarations or Runtime-scoped configuration.
References use the exact `CapabilityRef("capability", "documents", 2)` above.
The SDK alone owns composition, activation, `for_run`, and cleanup. Tool results
are projected once after a raw success, before native success observers; an error
recovered by an SDK hook is not a raw success. Projection failure is not permission
to repeat an already-completed tool. Runtime reuses these callbacks at its original
pre-receipt boundary and does not install a second native projection path.

Referenced input is loaded only into an ephemeral provider view. History keeps
the descriptor and summary, while provider errors and retained attempt errors
omit input bodies. Built-in `ModelBinding.materialize()` models carry their input
contract. A custom materialized model must supply the public `ModelInputValidator`
contract; the open `ModelBinding` protocol alone does not prove input permission.
An unknown actual model or an unsupported native fallback composition fails before
loading or provider IO when references are present. Ordinary capabilities without
references remain usable with ordinary native models. Model overrides do not borrow
the previous model's permissions. Explicit `ReferencedInputsCapability.load()`
remains available for authorized reads and integrity checks.

A bound Workspace enforces its own `WorkspacePolicy` for native tool calls.
`ask` uses the SDK's `DeferredToolRequests` / `DeferredToolResults` approval flow;
callers must include deferred requests in the Agent output type when needed.
`deny` rejects before a sandbox operation and does not affect unrelated business
tools. A custom `wrap_toolset` takes responsibility for the complete execution
boundary, including authorization; Runtime supplies its admitted boundary there
instead of applying both native and Runtime policy paths. Workspace policy does
not turn an unrelated native MCP transport into an OS sandbox or authorize all
other file-reading tools.

The creator retains resource ownership. Public standalone adapters borrow their
sources, delegates, and open sessions; they do not close those dependencies.
Runtime drains its own work and closes its owned services without taking ownership
of application clients. A reference loader cleans up temporary resources it creates
on success, failure, and cancellation. Neither mode duplicates the SDK's
`for_agent`/`for_run`, ordering, activation, or run-local cleanup lifecycle.

### Tool results before persistence

Use an Agent-scoped `ToolResultsCapability` to project successful tool results
before Runtime stores an external-effect receipt. The same hook covers
ordinary Python tools, Workspace tools such as `read_file`, and MCP tools:

```python
from collections.abc import Mapping
from pydantic_ai.tools import ToolDefinition
from linktools.ai.capability import (
    AgentContext, ToolResultsCapability, TOOL_MCP_SELECTOR_METADATA_KEY,
)

class ApplicationResults(ToolResultsCapability):
    async def process_tool_result(
        self, context: AgentContext[object], tool: ToolDefinition,
        arguments: Mapping[str, object], result: object,
    ) -> object:
        selector = (tool.metadata or {}).get(TOOL_MCP_SELECTOR_METADATA_KEY, tool.name)
        # Route explicitly supported results using selector and the admitted
        # context. Return result unchanged for all other tools.
        return result

features.capability(ApplicationResults(), id="application.results", revision=1, scope="agent")
```

For MCP, Runtime sets the metadata selector from the actual server binding and
upstream tool name; server-supplied metadata cannot override it. The selector
uses the existing `mcp_tool_selector(server_id, tool_name)` format. Other tools
use their model-visible `tool.name`. The context preserves the execution's
admitted principal. Arguments are a read-only top-level mapping of the existing
validated Python values, not another JSON conversion or authorization boundary;
processors must not mutate nested borrowed values or the tool definition.

Runtime first authorizes and invokes the tool, then applies selected result
capabilities in their captured capability/container order. Only the final
projection reaches the existing operation owner. Effect-free tools also run
the hook once but acquire no operation journal. Native SDK
`after_tool_execute` still runs once afterward; it does not retroactively
change an already stored operation receipt. Completed-operation replay and
offline response fixtures already supply final results and skip these hooks.

The default implementation preserves the result. An application can explicitly
return `ToolReturn(return_value=summary, content=[referenced_content])` for
reconstructible text or binary content instead of persisting its body. Its selected
`ReferencedInputsCapability` must independently authorize and rebuild the
exact immutable content on consumption and recovery. A summary or temporary
path alone cannot restore bytes. Keep necessary receipts and non-reconstructible
results; this is not automatic lossy summarization of every tool result.

Use `ReferencedContent(..., representation="text")` for content that should
reach the model as an ordinary string, without attachment-profile permission.
The loader still returns the exact version's bytes; Runtime verifies their
size and digest before strict UTF-8 decoding. MIME remains descriptive metadata
(for example, `application/json` or `text/csv`), not a representation selector.
The default `representation="binary"` retains existing file-input MIME
admission and SDK encoding, including for `text/plain` documents. Invalid UTF-8
fails before provider dispatch; bytes are never silently replaced or repaired.
Both representations keep the existing input-size limits and transient-content
lifecycle: only descriptors/consumption summaries are retained, not loaded bodies.

Several selected `ReferencedInputsCapability` contributions may serve different
source references. Runtime routes each descriptor only to its exact selected
`CapabilityRef`, retaining the original actor and saved revision. Missing sources
fail before any loader runs; a single contribution containing multiple loaders
is ambiguous and is rejected. There is no fallback to a different source or a
second loader registry. Reopened hosts must still supply the saved capabilities
and immutable resource versions.

Optional `source_parameters={...}` retains source-owned JSON reconstruction
arguments with a reference. Values are detached and deeply immutable; they do
not enter the displayed model summary or define capability identity. Keep only
necessary parameters, never content bodies or credentials. The selected loader
must still authorize the original actor and exact parent/version; parameters
are not authorization proof. Keep `AssetKey.id` short and opaque instead of
encoding a complete reconstruction request into it.

File-result projection preserves existing `files` receipts by copying each bound
binary's `identifier` to its replacement reference. Runtime supplies the original
occurrences before processing and validates their identifiers, MIME, size and
digest afterwards. Equal content alone never identifies a file occurrence.
The optional reference `identifier` is not source identity or authorization;
ordinary text references need no file occurrence.

The local context heuristic accounts for pending text references using their
UTF-8 byte size, approximately four bytes per token, rather than only their
small descriptor. This is an estimate without loading content; actual provider
usage and explicit provider token counting remain authoritative.

Result and input callbacks borrow the effective instances returned by native
SDK `for_run`, including initialization and cleanup on those same instances.
Runtime does not bind or close them again. Result dispatch follows native
capability ordering and activation, with isolated scopes for concurrent calls.

Result capabilities use normal capability `id + revision` capture and recovery,
including Asset-loader contributions. Reopened hosts supply the implementation
for the saved revision; callers own same-revision consistency and bump the
revision when changing behavior. There is no callback digest or second result
store. A processing error or cancellation after a tool returned cannot establish
that an external effect is safe to repeat: existing unknown-effect recovery
rules remain intact. Original tool errors, permissions and effect policies are
unchanged.

## 3. Workspace and declaration assets

Registering `WorkspaceCapability(workspace)` at Runtime scope contributes the stable
Workspace tools and sandbox boundary only. It does not discover Agent, Skill, or
MCP declarations from Workspace paths.

Agent, Skill, MCP, and repository rule files use one explicit source: pass a ready
`AssetStore` to `AssetCapability(store)` and register that capability with
`features.capability(assets)`. A store-backed group
captures the declaration metadata visible when capture begins. Assets added
afterward are ignored for that capture; assets actually read by a loader must
still match their captured metadata through verification. Conflicting
identities or layouts fail closed. Omitting `AssetCapability` contributes no Asset declarations.

```python
from linktools.ai.capability import AssetCapability
from linktools.ai.runtime import RuntimeFeatures

assets = AssetCapability(store)
features = RuntimeFeatures("application")
features.capability(assets)
capture = await features.capture()
```

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
kind with `assets.loader("audit", loader)`. One kind has exactly one loader;
registering `agent`, `skill`, `mcp`, or `rule` replaces that built-in
parser slot instead of chaining with it. Every loader receives the same
`CapabilityLoadContext`: `read()` / `read_many()` read the captured
immutable versions, while `bind_versions()` returns their `AssetVersionRef`
values for execution-bound resources. Loaders return normal
`CapabilityContribution` values, or `RepositoryInstructionDocument` for
instruction-only inputs such as Rule. Use
`CapabilityContribution.from_declaration(...)` for Agent, Skill, and MCP
declarations. Resource-backed declarations remain owned by the same
AssetCapability AssetStore; a custom loader may change declaration format or
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
`MCPServerSpec` with `RuntimeFeatures.mcp(server)` (or returns it from a
loader) into the same `RuntimeFeatures.capture()` path.
Supported transports are stdio, Streamable HTTP (`http` and
`streamable-http` authoring names), and explicit legacy `sse`; LinkTools
does not fall back from one remote transport to another.

`RuntimeFeatures.capture()` returns a `RuntimeFeaturesCapture`. Pass that
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
configuration values may change under that identity without a full-configuration
equality check. Callers own same-revision consistency. Stored resource references,
source authorization and the effective sandbox execution policy are still validated;
identity equality does not authorize access to different resources.

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
from linktools.ai import RuntimeFeatures
from linktools.ai.spec import MCPServerSpec, mcp_server_selector

group = RuntimeFeatures("services")
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
# Pass features=group to Runtime.open(...).
```

Import a shared configuration explicitly, then register its declarations:

```python
from pathlib import Path
from linktools.ai import RuntimeFeatures
from linktools.ai.spec import MCPServerSpecAdapter

group = RuntimeFeatures("services")
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
the same logical service, data domain, and behavior. Callers must use a new ID
or revision when changing logical behavior; LinkTools does not infer a revision
change by comparing configuration dictionaries. Tenant access, resource isolation
and execution policy are enforced independently. Asset capture does not freeze
a remote server's implementation.
Resources/prompts APIs, OAuth, dynamic end-user credentials, mTLS, server
publishing, and arbitrary third-party configuration import are out of scope.

### Execution sandbox

Workspace filesystem and shell tool effects run through the public `Sandbox` / `SandboxSession` boundary. Inject a custom implementation with `RuntimeFeatures(..., sandbox=...)`; register `WorkspaceCapability(workspace)` and configure Sandbox on that same RuntimeFeatures object. A Sandbox can also be configured without a Workspace for Skill resource paths and MCP stdio.

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
3.11. It has no automatic Local fallback. Bubblewrap isolates Session
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
from linktools.ai import RuntimeFeatures, Workspace
from linktools.ai.capability import WorkspaceCapability
from linktools.ai.workspace import LocalSandbox, ReadOnlySandboxPolicy

readonly = ReadOnlySandboxPolicy(
    readable_paths=("src/**", "README.md"),
    resource_paths={"review": ("**",)},
)
workspace = Workspace.load("/workspace/project")
group = RuntimeFeatures(
    "workspace",
    sandbox=LocalSandbox(read_policy=readonly),
)
group.capability(WorkspaceCapability(workspace))
```

When a `SandboxResource` declares `files`, file operations expose only those
members permitted by the read policy, and directory listings expose only their
permitted ancestors. A simultaneous `source` locates that set; it does not grant
the neighboring files. A source-only resource still declares the directory.
This constrains Sandbox file APIs, not trusted Local host processes. Applications
needing additional resource files must declare them explicitly. Independent
Workspace-relative grants retain their own policy.

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

Model-visible control text is partitioned by lifecycle. Binding-static Agent instructions, Skill catalog, and Workspace guidance form the most stable prefix. Execution-static capability guidance and the repository instructions already active when the Execution starts follow that prefix and remain fixed for that Execution. Repository sources first discovered later through declared workspace path fields are kept in one refreshable overlay; only that overlay changes when a new scope becomes effective. Provider adapters may serialize these instruction parts as `system`, `developer`, or another supported instruction channel, and the physical request may still carry the same fixed prefix on every model call so provider prompt caching can reuse it.

The fixed prefix is not appended as a new conversation message on each model call. Lazy Skill bodies/resources, Memory contents, the current-plan reminder, user input, tool and Subagent results, attachments, and retry/error feedback remain dynamic request or conversation context. Memory guidance is fixed only for an Execution that enables Memory; Memory contents stay pull-based through the Memory tools, so updates from another Execution are observed when the Agent reads/searches Memory again rather than being pushed into the standing prompt. Workspace instruction text describes stable usage constraints only; authorization and approval remain Runtime-enforced behavior. Recorded requests may therefore contain the same fixed instruction parts repeatedly without representing repeated historical messages.

`allow_tools` controls ordinary/external model-visible tools. `mode="plan"` selects plan mode for an execution; `planning` controls whether the Agent has the planning capability. Plan mode implies `planning=True`. Runtime infrastructure capabilities such as planning, memory, Skill loading, and Subagent delegation are composed by Runtime according to the resolved execution contract.

Subagents are root Agent definitions selected from the same captured catalog. A root Agent cannot select itself as a Subagent, and the Runtime does not create a second registration system for child Agents.

## 5. Output contracts

The resolved output contract belongs to an execution, not to Session identity.
An Agent may declare a default JSON Schema through `AgentSpec.output_schema`,
`RuntimeFeatures.agent(..., output_schema=...)`, or the `output_schema` field in
its Asset declaration:

```python
from pydantic import BaseModel

class ReviewResult(BaseModel):
    approved: bool
    findings: list[str]

application.agent(
    "reviewer",
    model="decision-route",
    output_schema=ReviewResult.model_json_schema(),
    output_retries=2,
)
```

Root runs, delegated children, and Agent-backed Tasks use that same default.
Omitting `output`, or passing `output=None`, inherits the declaration; an Agent
without a declared schema produces text. An explicit run output type overrides
the default, and `output=AssistantTextOutput` from `linktools.ai.agent` explicitly
selects text even for a structured Agent. Task nodes can similarly override the
default with `output_type`.

The schema is validated through the existing `OutputBinding` structured-output
path, including native Pydantic AI correction retries controlled by
`output_retries`. Only JSON Schema constraints are declared; Python-only model
validators are not persisted. Each execution freezes the resolved output mode
and schema in its `AgentBindingContract`; recovery uses those stored values,
not a later Agent declaration.

An explicit per-run contract can be selected as follows:

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
- the resolved model configuration as diagnostic metadata (identity is the registered model ID);
- the selected capability pins, including bound Skill Asset version references when Execution state is durable;
- selected Subagent ids and their direct execution bindings;
- `output_mode`;
- the canonical output JSON Schema;
- one `binding_digest`.

The binding contract does not persist Python output import paths or duplicate identity hashes. `ExecutionResult` exposes the derived `output_contract_digest` together with the terminal output.

## 6. Sessions and executions

Session metadata and a first turn are available through the default Session capability.
An unretained completed turn makes its Session context unavailable for later
resume or fork. To retain model context across resumed turns, register `ExecutionCapability(retain_history=True)` at Runtime scope before
opening the Runtime:

```python
from linktools.ai.capability import ExecutionCapability

application.capability(ExecutionCapability(retain_history=True))
# Open with Runtime.open(..., features=application) after registration.
```

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
use `max_retries=2` with `retry_delay=1.0` as the initial exponential backoff,
bounded jitter and provider Retry-After hints; see
[transport retry semantics](docs/model-protocols.md#transport-retries).
Agent `tool_retries` and `output_retries` default to `None`.
`RuntimeFeatures` supplies `3` for both unless configured otherwise; an explicit
Agent value, including `0`, wins. The compiler resolves retries before durable
binding, so recovery retains the admitted values across Runtime reconfiguration.

Model attachments use one string list, `input_types`, which defaults to empty.
Plain prompt text is unaffected; there is no `text` category. The built-in names
`image`, `audio`, `document`, and `video` expand to their corresponding Pydantic AI
MIME Literal values. Use a `mime:` prefix for an explicit MIME type or `type/*`
family. Attachments that do not match the resulting MIME allowlist are rejected:

```python
from linktools.ai.model import create_model

binding = create_model(protocol="openai",
    model="your-model",
    input_types=["audio", "document", "mime:audio/*", "mime:application/pdf"],
)
```

For `lt ai run`, `lt ai acp`, and `lt ai web`, the existing
`.linktools/config.yaml` uses the same strings:

```yaml
model: your-model
input_types:
  - audio
  - document
  - mime:audio/*
  - mime:application/pdf
```

The model package directly re-exports Pydantic AI's `AudioMediaType`,
`ImageMediaType`, `DocumentMediaType`, and `VideoMediaType` Literal types with
their SDK names and identity. Category expansion uses those SDK sets, rather
than treating a category as a wildcard. For example, `audio` means the SDK's
known audio MIME values; `mime:audio/*` permits the whole audio family.
Custom values such as `mime:application/x-private` remain supported. Bare MIME
strings, `mime:*/*`, and partial wildcards are not accepted; actual attachments
still need concrete MIME types. Text files can be selected explicitly, for
example with `mime:text/plain`, without changing ordinary prompt text handling.

Capture expands categories, normalizes, deduplicates, and sorts the MIME values.
Model diagnostic metadata stores the resolved `input_mime_types` list, not
model identity. Restore resolves the current binding by registered ID and validates
new inputs against its current permissions and SDK support. Callers own consistency
when changing configuration. `input_mime_types` is the diagnostic/display field,
not the Python or YAML authoring argument. Web settings show the current resolved
MIME values rather than authoring category labels.

The allowlist declares application permission, not guaranteed provider support.
SDK encoding restrictions and existing resource limits still apply. Transport
selection is internal; no extra status, transport, passthrough, or per-MIME
byte-limit configuration is needed.

Configure attachment permissions with `input_types`. Captured model contracts
must contain the resolved `input_mime_types` list; missing permissions do not
implicitly allow attachments. Configuration changes under the same registered ID
do not change model identity or block restoration; current attachment checks still apply.

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
        mime_type="text/plain",
        identifier="report-1",
    )),
)).result
```

The Sandbox canonicalizes logical paths before reading them and preserves every input occurrence. Passing the same path twice therefore produces two attachment occurrences with distinct execution-local `attachment_id` values, while their content digests may be identical. The initial model request receives each file as `BinaryContent` together with its canonical Workspace path, and the captured bytes are recovered from Runtime storage rather than reread from the Workspace during retry or recovery. After a complete model response consumes that binary input, Runtime keeps only lightweight file/path context in the active model context, so later agent-loop requests, Session turns, and forks do not repeatedly resend the bytes. The raw transcript remains lossless.

For new Agent configurations, select `read_file` to preview text or attach supported files (`read_file(path=[...])`; use `as_file=True` for whole-text input). See [file tools and model input](docs/file-tools.md) for format capabilities, bounded reads, exact edits, minimal toolsets and migration. The legacy `attach_files(paths=[...])` entry point remains a normal `filesystem.read` Workspace tool: it applies the existing Sandbox, path, approval, and repository-instruction boundaries, preserves duplicate occurrences, reads the current Workspace contents, and sends those files only to the next model request. Runtime binds those occurrences to the originating tool call before the content enters model history, so parallel tool calls do not require transcript-order inference. A later complete model response consumes them under the same transient rule. Agent-backed Task nodes use `runtime.tasks.from_agent(...)` and `AgentTaskInput`; place workspace paths in its `files` field and prompt content in `prompt`. Runtime materializes those inputs at graph admission, or when accepting a dynamically expanded batch, before dependent nodes run. Replaying an accepted graph does not reread the source files; graph state retains the captured input objects. Delegated subagents use the same explicit execution-input rules.

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

`TaskNode.timeout_seconds` also limits Agent-backed nodes. The interval starts at
that execution's first durable start and covers its model, tool, and delegated
subagent work. Recovery derives the same absolute deadline from the stored node
and original start time; it never grants a fresh interval or dispatches expired
work. This is an execution limit, separate from an HTTP timeout or a caller's
`run.wait(timeout_seconds=...)` observation limit. Applications can map their own
configuration into `TaskNode.timeout_seconds`; Runtime does not interpret business
metadata.

A graph's `RunBudget.deadline_at` can shorten the node deadline. The earlier limit
wins (the graph deadline wins a tie), without replacing the graph's shared model,
tool, or token budget. On expiry, Runtime requests execution-tree cancellation
and reads back its outcome. Confirmed cancellation yields a failed node with
`EXECUTION_WAIT_TIMEOUT` (or `EXECUTION_USAGE_LIMIT_EXCEEDED` for the graph limit),
while the underlying execution remains cancelled. Unconfirmed cancellation or
external effects stay `RECOVERY_REQUIRED`; cancellation settlement has a separate
five-second bound. A result already completed within its deadline remains valid
when read later. Neither an absent node timeout nor an absent graph deadline adds
an execution deadline. Automatic recovery checks the same deadline before
relaunching an execution, including deadlines inherited from Task-bound ancestors.
Missing invocation evidence fails closed rather than granting fresh time.

### Runtime context and execution queries

`Runtime.open()` is the stable Runtime construction entry point. Use
`RuntimeContext` to provide the application object, tenant, correlation, and
low-cardinality metric dimensions for the Runtime lifetime:

```python
from linktools.ai.runtime import RuntimeContext

context = RuntimeContext(app, tenant_id="tenant-a")
async with Runtime.open(
    "web-chat",
    storage=storage,
    features=features,
    context=context,
) as runtime:
    ...
```

`await runtime.close()` retains its default cancellation semantics. For a planned
worker restart, use `await runtime.handoff()` before releasing the application's
owner lease. This is a terminal lifecycle operation, not a pause of the same
Runtime. It stops admission, joins admitted starts, drains in-flight tool
settlement, and stops local producers without turning infrastructure shutdown
into user cancellation. Successful handoff closes owned resources; a later
`close()` or context-manager exit does not cancel the preserved executions.
Cancellation of the handoff caller does not abandon its owned cleanup task.
Retry `handoff()` or `close()` to join that same cleanup after a failure.

Handoff requires `CheckpointCapability`, synchronous execution admission, and
durable execution, recovery and checkpoint storage. Unsupported composition
raises `CAPABILITY_REQUIRED_MISSING` before closing the Runtime. Active evaluation
trial coordination is not supported by this lifecycle operation and is rejected
before shutdown; finish or explicitly close those trials first. A handoff cannot
replace an already-started default close and reports an error instead of claiming
that cancelled executions were preserved.

A fresh Runtime using the same namespace, tenant, stored objects and compatible
captured definitions resumes through the existing public control operation:

```python
handle = await replacement.executions.recover(
    execution_id,
    principal=authorized_actor,
    idempotency_key="known-owner-handoff",
)
```

An already-open replacement can also call `executions.recover()` for a
`FINALIZING` execution with a persisted terminal handoff. Recovery finishes that
prepared result publication without launching a new model/tool producer or
changing its generation. Repeated keyed recovery converges on the same result. An unprepared
`PENDING_START` without a durable checkpoint reports `EXECUTION_NOT_READY`;
recovery does not fabricate its missing input or an execution checkpoint.

The execution ID, admitted principal and session binding remain unchanged.
Completed tool results retain their operation-ledger identity; unknown external
effects remain `RECOVERY_REQUIRED` and must not be blindly replayed. Pending user
cancellation retains precedence. New control requests on the closing Runtime
are rejected; another Runtime can use the existing durable cancellation APIs.
Applications still own liveness/lease policy before takeover. Handoff does not
provide a network discovery service, shared filesystem, or replicated Workspace.
Filesystem and local SQLite storage require access to their original data;
cross-node recovery requires shared durable SQL and object storage. A tool that
cannot settle can keep handoff pending: a successful return never means that a
still-running local side effect has been abandoned.

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
history projections without loading model bindings or compiling Agents. Pass the
same `features=` declaration or capture when capabilities configure custom storage
routes or caches; read-only opening does not activate hooks or metrics recording.
Use `list_sessions(principal=..., cursor=..., limit=...)` for a paginated session
list. `messages(session_id, principal=..., cursor=..., limit=...)` returns an ordered
`Page[ModelMessage]` from a fixed committed history prefix, including system,
user, assistant, tool and attachment parts. It does not filter messages into UI
text, include uncommitted staging, or inherit a fork parent's later suffix.
The reader resolves stored content references in batches rather than asking the
caller to join Trace, Execution and object storage.

`session_detail(session_id, principal=..., max_age=0)` returns a coherent
`SessionDetail` containing session metadata and timeline entries, including
canonical structured output and authorized pending-turn coordination. Set
`max_age` to reuse the shared session-display cache policy. To inspect status
without loading usage, use `inspect_execution(..., include_usage=False)`;
configured coordination also supplies pending executions. These reads do not
initialize models or assets or activate recovery writes.

`inspect_execution()` returns safe durable summaries for binding, input,
output, timestamps, usage, and errors; it does not return prompt/output bodies
or raw error diagnostics. Its optional `ExecutionInfo.capability_refs` contains
immutable `linktools.ai.spec.CapabilityRef(kind, id, revision)` values from that
execution's frozen Agent binding: selected tools, Skills, MCP servers, runtime
capabilities and directly allowed child Agents (`kind="agent"`). The main Agent
identity remains `agent_id`. An Agent with no selected capabilities returns `()`;
non-Agent executions return `None`. `recent_executions()` exposes the same
projection after authorizing each execution. This is the execution's bound
capability set, not the current catalog and not evidence of an actual tool call,
child execution, or Skill body read. MCP entries identify bound servers, not
their expanded remote tool lists; built-in helper tools are not a separate
catalog in this field. No descriptions, prompts, resource paths,
MCP arguments or model configuration are included, and projection adds no
repository or object reads. Existing `ExecutionInfo` constructors may omit the
new field; it defaults to `None`.

The same capability identity projection belongs to
`AgentBindingContract.capability_refs`, so it is usable before persistence and
after restore, without a Runtime or Trace UI. History delegates to this binding
property rather than defining a second selection rule:

```python
binding = compiler.bind(compiled_agent, output=None).binding_contract
refs = binding.capability_refs
restored = AgentBindingContract.from_payload(binding.to_payload())
assert restored.capability_refs == refs
info = await runtime.history.inspect_execution(execution_id, principal=principal)
# info.capability_refs describes this execution's binding, including derive overrides.
```

The projection preserves selected capability order and then direct child Agent
references. It does not recursively include a child's selected capabilities or
replace the separate main Agent/Task identity. A compiled Agent must first be
bound to resolve exact direct child revisions; inspection never guesses them
from a current catalog. TaskGraph Agent nodes use their own execution bindings;
application-owned Tasks do not acquire inferred Agent capabilities.

`result()`, `history()/trace()`,
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
reads only that execution. Retained model interactions and aggregate `usage()`
read committed active identities; explicit TRANSIENT routes use process-local
history. Running and interrupted requests retain unknown usage as unknown. See the
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
both policies and an explicit input-semantics discriminator on the single
`task_node` wire type. Generic Task inputs remain complete JSON; only Runtime-resolved
Agent nodes project prompt intent independently of its stored representation.
Superseded development shapes, including nodes without policy fields or the
input-semantics discriminator, are rejected rather than guessed. Existing unfinished
development tasks may need to be submitted again; this change does not migrate them. Published-version fixtures and readers are added only when a real
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
    storage=storage,
    features=features,
) as runtime:
    ...
```

`RuntimeStorage` supports in-memory, filesystem, SQLite, and SQL backends. Repository domains keep their ownership, transaction, recovery, and retention rules; `Runtime.open()` consumes the storage composition directly. Use the same `features=` declaration or capture for `Runtime.open`, `RuntimeHistory.open` and `RuntimeSnapshot.create` when configuring module routes. Offline export requires a caller-owned `SnapshotExclusiveGuard` that quiesces related writers and object cleanup; a read-only `RuntimeStorage` handle alone is not that boundary. The supported archive flow is: quiesce writers and object cleanup, export through a read-only `RuntimeStorage`, restore into an empty staging root, verify required history and object references, then let the application publish that staging root. `restore_snapshot()` restores data but is not itself an atomic publication primitive. RuntimeStorage snapshots include Runtime-owned objects only. Skill and MCP resource bytes remain Asset-owned and are persisted as Asset version references in
binding metadata; portable RuntimeStorage restore therefore requires the
corresponding Asset history to remain available through the AssetStore supplied
when the Runtime is reopened.

SQLite-backed `RuntimeStorage` supports the built-in durable TaskGraph scheduler without a SQLite-specific launcher or an external lock. Normal internal Task optimistic-CAS races are reread and converged by the Task domain. Durable ToolOperation terminal persistence is also lease-aware: a same-lease heartbeat racing terminal persistence is reconciled without replaying the tool effect. Genuine ownership, fence, idempotency, tool-result, effect-unknown, integrity, and storage errors remain observable. A newly created local SQLite database initializes its Runtime and `ai_objects` schema; an existing SQLite database is opened without automatic schema validation and is never implicitly migrated or repaired. When `object_store` is omitted, durable SQLite Runtime objects are stored in the same database through the built-in `ai_objects` and `ai_object_chunks` tables. An explicitly supplied ObjectStore remains available when object payloads should live outside SQLite. External SQL backends still require explicit schema provisioning/migration. Process workers must initialize their own Runtime and SQL engine inside the worker process; initialized Runtime, engine, session, or connection objects must not be reused after `fork()`.

SQL Runtime, Asset, and Metrics initialization and ordinary reads do not inspect
table schemas by default. Schema diagnostics are an explicit deployment or
maintenance operation; without them, an incompatible database may fail at its
first affected query. Normal persisted-data, ownership, and integrity validation
still applies. To check a database provisioned with the complete schema:

```python
from linktools.ai.migrate import build_sql_schema_metadata
from linktools.ai.storage import validate_sql

await validate_sql(engine, build_sql_schema_metadata())
```

For a database provisioned only for selected Runtime domains, use the same
public domain selection as provisioning; include ObjectStore tables when the
Runtime uses its default SQL ObjectStore:

```python
from linktools.ai.runtime import RuntimeDomain
from linktools.ai.runtime.state.schema import build_runtime_sql_metadata
from linktools.ai.storage import build_object_sql_metadata, validate_sql

metadata = build_runtime_sql_metadata(frozenset({
    RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION,
}))
build_object_sql_metadata(metadata=metadata)
await validate_sql(engine, metadata)
```

Asset-only diagnostics similarly use the public `build_asset_sql_metadata()`
plus `build_object_sql_metadata()` when the default SQL ObjectStore is used.
Metrics already provides `validate_metrics_database(engine)` and
`validate_metrics_sqlite(path)` in `linktools.ai.migrate`; neither runs on normal
store access. Explicit `provision_*` calls still create and validate their
requested schema. The existing explicit
`SqlStorageContext.initialize(metadata=...)` also remains supported. No schema
configuration flag is required.

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
    RuntimeFeatures,
    Execution,
    Runtime,
    Session,
    Workspace,
)
```

Package-specific public contracts remain available from their owning packages,
for example `linktools.ai.asset`, `linktools.ai.model`, `linktools.ai.spec`,
`linktools.ai.capability`, `linktools.ai.workspace`, and `linktools.ai.runtime`.
`RuntimeFeatures` and `RuntimeFeaturesCapture` are owned by `linktools.ai.runtime`;
`RuntimeFeatures` is also available from the top-level composition API.
Other package contracts include `AgentSpecAdapter`, `SkillSpecAdapter`, `MCPServerSpecAdapter`, `AgentDeclarationLoader`,
MCP selector helpers, `WorkspaceToolDeclaration`, and the
optional stdio sandbox protocols. `ErrorDiagnostics` is available from
`linktools.ai.errors`; its `to_payload()` includes collected sanitized exception
chains and stack traces without display truncation. See
[complete failure diagnostics](docs/runtime-history.md#read-complete-failure-diagnostics)
for authoritative reads, unknown historical fields and credential filtering.

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
