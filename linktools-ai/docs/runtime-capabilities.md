# Runtime-wide capabilities

`CapabilityGroup.capability(capability, scope="runtime")` registers a native
Pydantic AI `AbstractCapability[AgentContext[AppT]]` as host policy. The same
`capabilities=(group,)` argument to `Runtime.open` supplies both Agent-selected
behavior and Runtime-wide policy; there is no separate registration system.

```python
from linktools.ai.capability import CapabilityGroup
from linktools.ai.runtime import Runtime

# GlobalRateLimit is owned by the application; see the executable example below.
quota = GlobalRateLimit(
    quota_key="provider-project-a", rpm=60, tpm=60_000, token_reservation=4_000,
)
group = CapabilityGroup("application")
group.capability(quota, id="host.rate-limit", scope="runtime")
group.agent("default", allow_capabilities=())

capture = await group.capture()
async with Runtime.open(
    "application", models=models, storage=storage, capabilities=(capture,),
) as runtime:
    ...
```

The default `scope="agent"` is unchanged: `allow_capabilities` and
`capability_filter` choose the capability, and its versioned contract participates
in Agent bindings. Runtime-scoped policies bypass both selectors. They are
applied to direct Agent runs, Sessions, delegated Agents, Agent-backed Tasks and
resumed/recovered runs. Plain Python Task bodies and external SDK calls do not
pass through these hooks.

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
- Native hook ordering, `for_agent` and `for_run` remain in effect. Instances are
  shared by default; per-run copies should retain the shared backend. Acquire
  owned resources in `wrap_run`, not `for_run`. Runtime does not close an
  externally owned backend. Host lifecycle replacements must preserve the
  nondeferred, unique and nonreserved native-ID constraints; registration
  validation does not police arbitrary replacement instances.

## RPM and TPM without a backend dependency

The [executable example](../examples/runtime_rate_limits.py) implements an
application-owned `GlobalRateLimit` with a local asynchronous rolling window.
It requires no Redis package, ModelRegistry configuration or LinkTools backend
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
sandbox boundary. Use managed `CapabilityGroup` tools/MCP when LinkTools
selection, effect tracking, authorization and sandbox lifecycle are required.

See the native [custom capability documentation](https://ai.pydantic.dev/capabilities/custom/)
for public hook signatures and lifecycle. Provider quota details are documented
by [OpenAI](https://developers.openai.com/api/docs/guides/rate-limits) and
[Anthropic](https://platform.claude.com/docs/en/api/rate-limits).
