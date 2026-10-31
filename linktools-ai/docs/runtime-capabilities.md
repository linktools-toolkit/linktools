# Runtime-wide capabilities

`RuntimeFeatures.capability(capability)` registers a native
Pydantic AI `AbstractCapability[AgentContext[AppT]]` as host policy. The same
`features=group` argument to `Runtime.open` supplies both Agent-selected
behavior and Runtime-wide policy; there is no separate registration system.

```python
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import Persistence

# GlobalRateLimit is owned by the application; see the executable example below.
quota = GlobalRateLimit(
    quota_key="provider-project-a", rpm=60, tpm=60_000, token_reservation=4_000,
)
group = RuntimeFeatures("application")
group.capability(quota, id="host.rate-limit")
group.model("default", binding)
group.agent("default", allow_capabilities=())

capture = await group.capture()
async with Runtime.open("application", storage=storage,
    features=capture,
) as runtime:
    ...
```

Use explicit `scope="agent"` for Agent-selectable, versioned behavior:
`allow_capabilities` and `RuntimeFeatures.filter` select the capability, and its
contract participates in Agent bindings. The default Runtime scope bypasses
both selectors. Runtime-scoped policies are
applied to direct Agent runs, Sessions, delegated Agents, Agent-backed Tasks and
resumed/recovered runs. Plain Python Task bodies and external SDK calls do not
pass through these hooks.

## Capability and service ownership

All public capability declarations are imported from `linktools.ai.capability`,
including `ExecutionCapability`, `SessionCapability`,
`CheckpointCapability`, and `TraceCapability`.
`linktools.ai.runtime` owns Runtime entry points, services, and the
`RuntimeFeatures` / `RuntimeFeaturesCapture` composition contracts; it does not
provide duplicate capability exports.

A captured declaration can be reused across Runtime instances. Runtime binds
isolated `ConversationService`, `CheckpointService` and `TraceService` instances
for each run without changing the declaration or its custom native hooks.
`MetricsService` owns one producer recording lifetime over a borrowed
`MetricRecorderProvider` and enriches observations without creating a second
queue. Agent and each independent TaskEngine own separate recorders; Evaluation
borrows its TaskEngine.
Conversation, Checkpoint and Trace persistence owners keep their existing
module-specific reads, buffers and commit boundaries.

Module storage routes are validated during Runtime or read-only assembly,
before storage initialization. Invalid route values do not start storage I/O.
Session identity and combined execution-history queries remain owned by
`SessionService` and `ExecutionHistoryService` respectively.

## Optional execution persistence

Retained execution history, interrupted-execution recovery and detailed trace
are separate, explicit Runtime-scoped options. Passing durable storage alone does not
enable them. Automatic metrics use the independent `Runtime.open(metrics=...)`
resource described below. Supply one `RuntimeFeatures` declaration or
its `RuntimeFeaturesCapture` through `features`; complete registrations
before capture or opening the Runtime.

```python
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.capability import ExecutionCapability, CheckpointCapability, TraceCapability

chat = RuntimeFeatures("chat")
chat.agent("default", model="default")
chat.capability(ExecutionCapability(retain_history=True))
# Runtime.open(..., features=chat) retains model context and supports execution forks.

persistent_work = RuntimeFeatures("persistent-work")
persistent_work.agent("default", model="default")
persistent_work.capability(CheckpointCapability())
persistent_work.capability(TraceCapability())
# Runtime.open(..., features=persistent_work) supports recovery and audit.
```

`ExecutionCapability(retain_history=True)` retains committed model context for future turns and
execution forks. Session creation, listing, resume, metadata, and display history
are enabled by the default `SessionCapability`, independently of Conversation.
Register `SessionCapability(enabled=False)` at Runtime scope to disable Session
operations explicitly; no manual enable registration is needed by default. Resume still
requires available committed context: an unretained completed turn makes Session
history unavailable, and later resume or Session fork fails with
`SESSION_HISTORY_UNAVAILABLE` rather than silently losing context. Enabling
History retention enabled later cannot reconstruct that missing history. Execution forks
require an admitted conversation guarantee; checkpoint or trace data alone does
not grant it.
`CheckpointCapability()` retains before-request and after-node recovery context.
`TraceCapability()` retains execution transcript, step facts and model interactions;
trace alone cannot resume an interrupted run. Register the combination required
by the application. Each capability requires Runtime scope; duplicate registrations
of the same built-in feature are rejected.

Admission freezes the execution's conversation, checkpoint and trace guarantees.
Reopening with a different capability set does not upgrade, downgrade or
reinterpret that execution, including idempotent start replay. Runtime-wide
request policies, such as rate limits, still use current host configuration.
TaskEngine and Evaluations are Runtime-bound domains, not capability wrappers;
independent advanced composition contexts remain available. TaskEngine recovery
is explicit; binding or opening it does not launch pending graphs. Shared storage provisioning is independent of producer activation.

### Automatic Runtime metrics

`metrics=` accepts a `MetricRecorderProvider`, whose no-argument `recorder()`
creates an independent scope. A `Metrics` dataset naturally supplies a
synchronous recorder; a host provider can publish without a producer-side SQL
dataset. Supply it once to Runtime for its bound producers:

```python
async with Runtime.open("app", storage=storage,
    features=group, metrics=metrics,
) as runtime:
    engine = await runtime.tasks.bind(*tasks)
    ...
```

`Runtime.open(metrics=...)` records Agent observations and configures bound
Task engines to record Task observations, including evaluation execution.
Each producer owns its recorder and drain lifecycle; Runtime closes its bound
domains in dependency order without closing the caller-owned dataset, database
engine or cache. Independently opened Task engines accept `metrics=...` too.
`metrics=None` disables automatic recording; application calls on the dataset
remain available. Read-only history and snapshots never activate recording.
Host providers own their delivery policy; a conversation turn does not force
unrelated metrics delivery. See [provider ownership and migration](storage-caching.md#automatic-metrics-providers).

`TraceCapability()` retains structured execution evidence and model-request
metadata without original HTTP bodies. Opt into
`TraceCapability(record_http_bodies=True)` for transport diagnostics; original
request/response bodies can repeat whole prompts and contain sensitive content.
Checkpoint recovery keeps its structured context independently of this option.

### Module storage and cache configuration

`StorageRoute` and `StorageRetentionMode` are imported from `linktools.ai.storage`.
A route bundles its backend with an optional borrowed hot-read cache, for example
`StorageRoute.sqlite("state.db", cache=cache)`. Cache identity is not part of
persistence identity. Session and Conversation use the root `PersistenceStorage`
resource; `ExecutionCapability(retain_history=True)` only selects context retention. Set
`PersistencePlan(conversation=route, ...)` to customize that domain.
For plan-based construction, caches live on the routes rather than a second
`PersistenceStorage` cache argument. Convenience factories such as
`PersistenceStorage.sqlite(path, cache=cache)` apply the cache to their generated routes.

`PersistencePlan(checkpoint=route, trace=route, ...)` configures archive routes;
`CheckpointCapability()` and `TraceCapability(commit=...)` only configure behavior.
Omitted archive routes inherit the execution backend and cache
with separate logical partitions. An explicit route supplies the complete
resource: `cache=None` disables that route's cache instead of inheriting it.
Trace and Checkpoint routes must share the execution StateStorageGroup;
incompatible routes fail during opening. A separate root Conversation route
uses durable admission and recoverable prepared terminal handoffs rather than
single-transaction deferred turns. Shared SQL engines, caches, and object stores
remain borrowed resources.

The workspace ACP convenience entry point accepts the same execution declaration:
`ACPApplication.for_workspace(workspace, features=features, storage=storage,
execution=ExecutionCapability(retain_history=True))`. Supply coordination through
`PersistenceStorage(..., coordination=cache)` or its convenience factories. Without
an explicit execution declaration, ACP defaults to retaining history. An explicit
declaration controls retention and commit; workspace
capability remains built in. A cache attached only to storage does not enable
turn or stream coordination.

`PersistenceStorage(..., coordination=cache)` explicitly supplies shared turn
coordination and live streams. The execution route cache only serves hot
execution record reads. Session metadata/display and Conversation content
use the Conversation route cache; shortcut storage factories apply the supplied
hot cache to their routes. Pending-turn visibility comes through the explicit
coordination channel, not through the hot cache.
A module resource override cannot create a separate producer ownership
namespace. Pass the shared Persistence to `History.open(persistence=...)` and
`Snapshot.create(persistence=...)` to read and export all custom routes without
activating Agent hooks or Metrics recorders.
Restored portable generations use their local routes and caches.

Shared filesystem group manifests, locks and empty domain metadata are provisioned
independently of selected features to preserve atomic recovery and historical reads.
Unselected modules do not write new business archives or start observation workers.

### Retry defaults

`AgentSpec.tool_retries` and `AgentSpec.output_retries` default to `None`, as do
omitted fields in declaration payloads. `RuntimeFeatures` supplies defaults of
`3` for both; configure them through the namespaced defaults map:

```python
features = RuntimeFeatures("application", defaults={
    "agent": {"tool_retries": 5, "output_retries": 2},
    "custom.search": {"timeout": 10},
})
```

Custom namespaces are retained for their owning consumers. Agent retry defaults
are validated when the compiler consumes the `agent` namespace. Omitted keys
use the shared default of `3`; explicit Agent values, including `0`, win.
The compiler resolves these values before creating a durable binding. Captures
freeze the configured defaults, and restoring an execution uses its saved
resolved values even if a later Runtime has different defaults.

Drain active work and upgrade all Runtime readers and writers sharing a namespace
before admitting executions with the new defaults. Superseded pre-release record
shapes and mixed-version operation are not supported.

Without retained history, checkpoints or trace, ordinary execution keeps authorization, idempotency,
producer ownership, cancellation, terminal results and durable tool-effect
records. Model and tool progress still streams. Local history is available while
its existing staging owner is retained, and is unavailable after release or in
an independent reader. With Conversation enabled, read committed conversation through Session history.
Intentional approval/deferred waits retain the continuation needed to resume;
prepared cross-store terminal handoffs also retain their required evidence.
These boundaries do not enable arbitrary restart after a process crash.

`recover()` continues interrupted execution only when its admitted guarantee
includes restart checkpoints. Otherwise it raises `CAPABILITY_REQUIRED_MISSING`;
unresolved tool effects retain their stronger recovery diagnostic. It does not
silently convert recovery into cancellation. `auto_recover` still controls startup
reconciliation; it does not grant a missing execution guarantee.

For an ordinary Agent without restart checkpoints, explicit
`execution.cancel(force=True)` may safely revoke the producer after cancellation
has blocked new tool admission and all tool-effect claims are settled. It closes
the control head and commits cancellation before releasing the Session. A late
model response cannot overwrite that terminal result. This is explicit
revocation, not proof that a remote process or provider physically stopped.
Ordinary cancellation without force retains its owner-settlement behavior.
Live claims and unknown external effects keep cancellation unresolved; force
does not bypass explicit effect resolution. Delegated calls use the same authorized recursive cancellation path. Task-owned
execution trees retain their existing cancellation ownership.

Low-level implementations of the module-exported `ExecutionBackend` protocol
must accept `cancel(execution, *, force=False)`. An implementation that does not
support explicit revocation can ignore the keyword and keep returning its
existing confirmation outcome. It must not claim completion from force alone.

### Independent commit boundaries

Import `CommitPolicy` from `linktools.ai.core`. Its factories are
`CommitPolicy.sync()`, `CommitPolicy.terminal()`, and
`CommitPolicy.batch(buffer=..., batch_size=..., flush_interval=...)`. The policy
pairs write timing with its optional borrowed buffer, independently of read
cache routing. In batch mode without a buffer, observation writes await the source. A buffer
never becomes a hot-read cache and is never closed by the policy.
Runtime execution supports sync and terminal. Trace supports sync, batch and terminal;
Metrics datasets always write synchronously; a host provider separately chooses
automatic-delivery timing. Conversation has a fixed terminal boundary;
Checkpoint has fixed model/tool safety boundaries. They do not accept timing
preferences that would weaken those guarantees.
`ExecutionCapability(commit=CommitPolicy.sync())` is the default: new turns use durable admission even with
a coordination cache. Shared admission ownership and idempotency coordination
remain active for every policy, including a sync caller joining an already
admitted terminal-policy turn. Opt into `ExecutionCapability(commit=CommitPolicy.terminal())`
to defer eligible ordinary native turns until the terminal boundary. Deferred
admission additionally requires explicitly configured `coordination`
with the required coordination groups and compatible storage routes.
Execution timing is independent of Conversation retention and Session activation.
Execution is required and installed by default. Register one explicit
`ExecutionCapability` on the runtime scope to replace its default configuration:

```python
from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy

from linktools.ai.persistence import PersistenceStorage

storage = PersistenceStorage.sqlite("state.db", coordination=shared_cache)
features.capability(ExecutionCapability(
    commit=CommitPolicy.terminal(),
), scope="runtime")
```

`Runtime.open` does not take a separate execution commit policy.

Use `TraceCapability(commit=CommitPolicy.terminal())` to retain optional trace
observations in process-local staging until a required publication boundary. It
needs no write buffer and does not make pending facts durable or hot-readable.
When Trace, Conversation and Execution resolve to the same transaction resource,
the existing terminal transaction includes the trace tail and final conversation
state. Explicit flush and recovery/tool safety boundaries may publish earlier.

Enabling durable Trace still requires durable admission and producer fencing,
even with `ExecutionCapability(commit=CommitPolicy.terminal())`. The terminal
transaction is shared; the entire traced turn is not promised to use only one
transaction. This reduces intermediate optional projection transactions, not
necessarily SQL statement count or network round trips. Trace and Checkpoint
currently require the same transaction resource as Execution; incompatible
routes are rejected during initialization. No distributed transaction is implied.

```python
from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import CommitPolicy
from linktools.ai.persistence import Persistence, PersistenceStorage

features = RuntimeFeatures("traced-chat")
features.agent("default", model="default")
features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()))
features.capability(TraceCapability(commit=CommitPolicy.terminal()))
storage = PersistenceStorage.sqlite("chat.db")
# Open Persistence with storage, then Runtime with persistence and features.
# Required admission/fencing writes remain; the final trace tail shares the
# conversation/execution terminal transaction.
```

Conversation commits its final context, Session cursor, and corresponding
Execution outcome at the turn boundary. Checkpoint retains its recovery-critical
model/tool boundaries. Trace defaults to batch size 128 and a 3-second interval.
Metrics does not share that scheduler. Explicit flush,
completion, close, and required safety barriers remain authoritative. One module
does not force unrelated modules to flush at every boundary.

Before side effects, Task ownership, or another durable dependency requires
state, Runtime establishes the necessary durable facts. Authorization,
idempotency, producer fencing, tool-effect ownership, and unknown-commit readback
remain part of the execution core. A process crash cannot recover an uncommitted
turn merely because Conversation or Trace was enabled.

Multiple Runtime instances need shared turn coordination for provisional
live turns. Without injected coordination, or when its required groups are absent,
Runtime retains ordinary durable admission. It does not silently substitute
private coordination. Remote preview remains
provisional. Result completion and unknown-commit resolution use authoritative
storage, and a lost lease never permits two canonical terminal results.

## Capture, identity and lifecycle

- A capture exposes every registration in `capture.contributions`, including its
  `scope`. `AgentCompiler.filter_capabilities` returns only Agent candidates;
  Runtime-scoped policies never invoke the selection callback.
- The existing `id`, `revision`, and `config` registration arguments remain
  available. Anonymous native instances need an explicit registration `id`.
  Native IDs must not collide across runtime policies or selected capabilities,
  or use the reserved `linktools.` prefix. Native combined capabilities are
  supported, including anonymous leaves. Runtime policies must not defer loading.
- Registrations are captured once when Runtime opens, or earlier with an explicit
  capture. Later group registrations do not affect an open Runtime. Do not mutate
  native capability configuration while its capture is in use.
- Runtime policies are current host configuration, outside durable Agent binding
  identity. Reopening uses the newly supplied host policies while restoring
  historical Agent behavior. Put model-visible, versioned behavior in Agent scope.
- For per-run history retention, Checkpoint and Trace, use `ExecutionCapability`,
  `CheckpointCapability` and `TraceCapability` types in native `requires`, `wraps` and `wrapped_by` ordering declarations.
  References to an original instance do not survive per-run copies, as described
  by the native SDK ordering contract.
- Native hook ordering, `for_agent` and `for_run` remain in effect. Instances are
  shared by default; per-run copies should retain the shared backend. Acquire
  owned resources in `wrap_run`, not `for_run`. Runtime does not close an
  externally owned backend. Host lifecycle replacements must preserve the
  nondeferred, unique and nonreserved native-ID constraints; registration
  validation does not police arbitrary replacement instances.

## RPM and TPM without a backend dependency

The [executable example](../examples/runtime_rate_limits.py) implements an
application-owned `GlobalRateLimit` with a local asynchronous rolling window.
It requires no Redis package, model configuration or LinkTools backend
protocol. Run it with `python linktools-ai/examples/runtime_rate_limits.py` after
installing the package; its demonstration uses an offline native `FunctionModel`.

It admits a request only when both quotas have room:

- RPM reserves one logical request before calling the native handler
- TPM reserves a positive token allowance before dispatch. A host callback can
  compute the allowance from request/model/context metadata instead of a fixed
  value
- Admission is atomic across both limits. Cancellation while queued admits
  nothing; cancellation or failure after admission keeps the unknown token charge
- Observed usage can increase a reservation into debt. The example never refunds
  below its estimate, since a provider can charge against a requested output
  ceiling rather than the final output length
- Active-stream token reservations remain held. Completed/failed reservations
  remain charged for a further minute; this intentionally overestimates usage
  rather than expiring an unknown long-running request

Share one limiter instance to share its quota across Agents or Runtime instances
in the same event loop. A matching quota label alone does not share counters.
For multiple processes, supply an application-owned distributed implementation
with atomic quota admission, expiry and cancellation semantics. A semaphore only
limits concurrency; it does not enforce RPM or TPM.

The sample is a conservative workload quota, not an exact reproduction of a
provider's rate-limit algorithm. Set a reservation that covers the input, tool
schemas and expected maximum output, and enforce an appropriate output ceiling.
Underestimates can exceed TPM before usage is observed; recording debt cannot
undo that request. Provider limits may use different input/output/cache token
weights, model families, project/credential pools or output reservation formulas.
Choose the shared quota key explicitly according to the provider account policy.

## Native request and usage data

`wrap_model_request(ctx, request_context=..., handler=...)` receives public native
objects; no LinkTools adapter is needed:

- `ctx.deps`: principal, namespace, execution/session identifiers, application
  context and correlation data
- `request_context.model`: native `system`, `model_name` and model settings;
  `request_context.model_id` can be `None` when Runtime supplies a model instance
- `request_context.messages`, `model_settings`, and `model_request_parameters`:
  current messages, settings, tool definitions and instructions
- A successful handler result is a `ModelResponse` with per-response `usage`,
  including input/output and cache token counters. Run-context usage is cumulative
  and its update timing depends on the native lifecycle; it is not a replacement
  for request-level settlement. No public usage is guaranteed on errors or
  cancellation, and zero counters need not mean a free request

Native hooks can transform request settings/model or the response and its usage.
An estimate at wrapper entry is not guaranteed to describe final on-wire input;
coordinate the host's hook ordering and estimation policy. Native `count_tokens`
is model-dependent, may make a separate provider request and is not a universal
local tokenizer. The example never uses private native usage ledgers.

## Streaming, retries and cancellation

`wrap_model_request` remains active through stream consumption and closure.
Release concurrency permits in `finally` or an asynchronous context manager, but
do not refund a rate-window charge merely because a stream closed. Cancellation
of an acquisition must not release another request's permit. Waiting is
cancellable through Runtime cancellation/deadlines; `wait(timeout_seconds=...)`
alone only stops observing an execution.

The hook covers a logical Agent request. LinkTools transport retries and
provider-internal retries happen below it; compaction summary model calls bypass
it. A strict provider-attempt rate limit therefore needs provider/transport-level
integration. Call the native handler once; do not retry it in a loop.

`wrap_tool_execute` is available on the same capability, but logical tool calls
are not necessarily new external effects or cache misses. Do not hold a shared
model permit across a whole run or a delegating tool while a child waits for the
same permit. Use separate pools or limit the actual leaf work.

Runtime-scoped capabilities remain trusted host Python, not an authorization or
sandbox boundary. Use managed `RuntimeFeatures` tools/MCP when LinkTools
selection, effect tracking, authorization and sandbox lifecycle are required.

See the native [custom capability documentation](https://ai.pydantic.dev/capabilities/custom/)
for public hook signatures and lifecycle. Provider quota details are documented
by [OpenAI](https://developers.openai.com/api/docs/guides/rate-limits) and
[Anthropic](https://platform.claude.com/docs/en/api/rate-limits).

### Observation batch timing

`TraceCapability(commit=CommitPolicy.batch(buffer=buffer, batch_size=128, flush_interval=3.0))` controls observation-only
flush pressure (counted in observation boundaries) and maximum scheduled delay
in seconds when `commit.buffer` supplies `append`, `read_batch` and `ack`. Each run owns
an independent cache-backed recording scope. A policy with `buffer=None`
awaits publication at each observation boundary; timing options do not create a
hidden Runtime queue. A lower threshold or interval can make partial trace
visible sooner at the cost of more writes.
Explicit flush, tool-effect, checkpoint and terminal safety boundaries can
publish earlier and cannot be delayed by these options. Cache configuration does
not weaken those boundaries.

Automatic Metrics uses `metrics=provider` independently of Trace scheduling.
A `Metrics` dataset provides synchronous recording; host providers choose their
own bounded delivery and consumer batching. There is no SDK Metrics timer or
shared Trace/Metrics threshold. Explicit flush, close and required safety
boundaries remain owner-local, and closing a recorder never closes its provider.

## Optional atomic persistence

Runtime-scoped native capabilities can opt into the structural
[`CapabilityPersistence` protocol](capability-persistence.md) to submit run-local
writes using `storage` and `commit`. Compatible participants share actual
transactions; the guide specifies failure, forced-boundary and recovery limits.
