# Model protocols and readable traces

Use Python >= 3.11 with `pydantic-ai-slim>=2.61.0,<3.0.0` and
`pydantic-ai-harness>=0.61.0`; Harness requires its matching SDK version.
These minimum versions provide the native Thinking semantics used here, including
`"max"`; no LinkTools provider capability table substitutes for SDK behavior.
The `evaluation` extra uses the SDK's `evals` extra to select the matching evaluation package.
Install only the provider adapters you use:

```shell
pip install "linktools-ai[openai]"       # Chat Completions and Responses
pip install "linktools-ai[anthropic]"    # Anthropic Messages
pip install "linktools-ai[openai,anthropic,web]"  # both adapters and Web console
```

The base package does not require either provider SDK. Core ModelBinding, Runtime
and Web modules remain importable without them (serving Web additionally needs the
`web` extra). Calling `create_model` for a provider without its extra fails immediately
with the matching install hint, before connection settings or credentials are resolved.

Choose the actual API, not a vendor-looking model name:

```python
from linktools.ai.model import create_model
from linktools.ai.runtime import Runtime, RuntimeFeatures, RuntimeStorage

features = RuntimeFeatures()
features.model("default", create_model(protocol="openai-responses", model="gpt-5"))
features.model("messages", create_model(protocol="anthropic", model="claude-sonnet-4-5"))
features.model("compatible", create_model(protocol="openai", model="gateway-model", model_settings={
    "extra_body": {"enable_thinking": True},
}))

async with Runtime.open("app", features=features, storage=RuntimeStorage.in_memory()) as runtime:
    ...
```

`create_model` returns a `ModelBinding`, not a registry. It accepts input types,
JSON-valued SDK model settings, credentials, timeouts and retry/token limits. Missing
provider extras fail with the matching installation hint. Unknown protocol strings
fail with `MODEL_CONFIG_INVALID` without echoing connection values.

The public type values are `openai` (Chat Completions), `openai-responses`, and
`anthropic` (Messages). CLI `--model-api`, `MODEL_API`, workspace `model_api`,
HTTP trace labels and input dispatch use the same values. Native HTTP endpoint
paths are unchanged. There is no separate public registry or `Runtime.open(models=...)`.

## Generic limits and configuration names

Keep connection/capacity options separate from request settings. For example:

```python
features.model("default", lambda: create_model(
    protocol="openai", model="gateway-model",
    context_window=131072, model_settings={"max_tokens": 8192, "timeout": 45},
    max_retries=3, retry_delay=5, retry_max_delay=120,
    retry_backoff_factor=4,
))
```

`context_window` is the total context capacity in tokens, not the maximum output
length or a provider capacity upgrade. A non-null explicit value overrides the
SDK's inferred context window and is recorded in `contract["context_window"]`.
`None` leaves the SDK default unchanged. Explicit values must be positive integers,
not booleans. SDK-inferred capability profiles remain internal to the adapter;
`create_model` does not expose a profile override.

`model_settings["max_tokens"]` reserves output tokens. Compaction uses a known context window minus
the effective output reserve and current instruction/tool/output-schema overhead.
If the SDK has no known window and no explicit compaction target exists, Runtime
does not invent one. Native fallback uses the smallest **known** candidate window;
unknown candidates do not constrain that value. This is context budgeting, not
up-front MIME validation of an unused candidate, and estimates do not
replace the provider's actual input limit.

The same seven numeric names are accepted in workspace `.linktools/config.yaml`:
`context_window`, `max_tokens`, `timeout`, `max_retries`, `retry_delay`,
`retry_max_delay`, `retry_backoff_factor`. The CLI adapter places `max_tokens` and
`timeout` in `model_settings` when constructing the binding; they are not duplicate
factory arguments. Configuration names use snake_case;
all three local commands use corresponding kebab-case flags such as
`--context-window` and `--retry-max-delay`. Existing LinkTools configuration uses
`MODEL_CONTEXT_WINDOW`, `MODEL_MAX_TOKENS`, `MODEL_TIMEOUT`, `MODEL_MAX_RETRIES`,
`MODEL_RETRY_DELAY`, `MODEL_RETRY_MAX_DELAY`, `MODEL_RETRY_BACKOFF_FACTOR`; the
standard environment prefix gives `LINKTOOLS_MODEL_CONTEXT_WINDOW`, etc.
Explicit CLI values win, followed by existing LinkTools configuration/environment,
workspace values and unchanged constructor defaults. Zero retries/delay remain
valid and are not mistaken for missing configuration.

Timeout and both delay values are seconds. Timeout is the provider HTTP timeout,
not the entire retry/run deadline; existing request/run deadline and cancellation
ownership remain unchanged. Retry count means additional attempts. Library defaults
remain timeout/output/context unspecified, two retries, initial delay 1, maximum
delay 60, and factor 2. The protocol value `openai-responses`, existing
`LINKTOOLS_MODEL_API`, and provider-native `OPENAI_*` / `ANTHROPIC_*` connection
variables retain their meanings. These generic options do not expose arbitrary
headers or `model_settings` JSON through the CLI.

For delayed configuration, register a synchronous zero-argument function returning
an existing `ModelBinding`:

```python
def load_assistant():
    config = load_configuration()
    return create_model(protocol="anthropic", model=config["model"],
                        base_url=config["base_url"], api_key=config["api_key"])

features.model("assistant", load_assistant)
```

The feature capture freezes route names and sources, not a callable's future
results. Each agent run resolves its own binding once and uses that same binding
for durable contract capture, input validation, tool rounds and model retries.
The next run calls the loader again, so updated configuration and credentials
apply without a second connection-resolver API. Concurrent runs are isolated.
Unused routes are not loaded; explicit metadata inspection may resolve a binding
without creating a provider client. A failed load can be retried; asynchronous
loaders and recursive same-route loads are rejected. A later feature registration
affects future captures only.

The registered route name is model identity. Recovery resolves the current binding
under that same route; protocol, provider model, request-body and connection changes do
not change identity or block task execution. Callers own configuration consistency.
Historical `model_contract` metadata describes the captured configuration for
diagnostics, not an equality requirement. Actual requests retain their own model
and protocol trace metadata. Agent, tool, output and execution ownership checks
remain unchanged. Static binding declarations reuse their declared configuration.

Model binding resolution does not create HTTP clients. `materialize()` creates
the client from the resolved binding's static settings. SDK-native environment
defaults still apply to omitted connection values.
The caller owns the returned model's async context and releases its client when
the run ends. Custom `ModelBinding` implementations still provide provider,
model_identity, contract and materialize; the route name belongs only to Features.
For diagnostics outside Runtime, use `async with binding.materialize() as model:`.

The earlier pre-release `openai-chat` / `anthropic-messages` authoring values are
not aliases; update configuration to the current spellings. Chat diagnostic
metadata omits the API field, interpreted as `openai`; Responses and Anthropic
include their explicit API. All three protocols share `create_model` options.
Protocol and provider model metadata describe a request, not the registered model
identity. A current binding may use a different protocol under the same route.

## Native decision models

Decision models answer bounded questions instead of generating free-form text.
Register an SDK-native `DecisionModel` through the existing public `ModelBinding`; no additional LinkTools registry or
protocol selector is required. `create_model` remains a convenience constructor
for its documented protocols, not an allowlist for Runtime models.

Use a Pydantic `BaseModel` as the execution's `output=` contract. Boolean fields,
string enums/`Literal` choices, and supported bounded numeric fields map to the
native SDK's questions. Do not use the default text output or assume arbitrary
free-form strings can be generated. The SDK owns supported schema/profile details
and decides whether a route requires a fallback. A returned class probability is
model output, not calibrated proof or action authorization.

```python
from pydantic import BaseModel, Field
from typing import Literal
from pydantic_ai.models import Model
from linktools.ai.core import JsonValue
from linktools.ai.runtime import RuntimeFeatures

class Decision(BaseModel):
    accepted: bool
    category: Literal["accept", "reject"]
    probability: float = Field(ge=0, le=1)

class DecisionBinding:
    provider = "application-decisions"
    model_identity = "evidence-decision"
    contract: dict[str, JsonValue] = {
        "provider": provider, "model": model_identity,
    }

    def materialize(self) -> Model:
        # Application-owned construction of the selected SDK-native adapter.
        return build_configured_decision_model()

features = RuntimeFeatures("decisions")
features.model("decision", lambda: DecisionBinding())
features.agent("classifier", model="decision", allow_tools=())
# Inside Runtime.open(..., features=features):
# result = await runtime.agents.get("classifier").run(prompt, output=Decision)
```

Install the chosen SDK adapter's own optional dependency only when needed. The
base LinkTools package neither imports a decision vendor nor adds one as a
mandatory dependency. Use an SDK version that provides the chosen native model;
older supported SDKs without decision models retain their existing functionality.
A loader is evaluated once per agent run/admission, not once per tool round or
individual model request; all requests inside that run share the captured binding.

Runtime's streaming observation works with the SDK's decision stream adapter.
That adapter can emit a completed structured response after a decision finishes;
this is not a promise of token-by-token transport streaming. Non-streaming SDK
requests use the same native schema semantics. Logical request usage and the
original response's `provider_details` (including available `confidence`,
`probabilities`, and `scores`) survive the existing durable history codec and are
read through `execution.model_interactions(include_content=True)` or the
principal-authorized Runtime executions facade. Missing values remain missing;
Runtime does not synthesize a distribution or reinterpret the SDK's confidence.

A custom binding owns its adapter's connection, transport retry policy, and SDK
fallback policy. It does not automatically acquire the built-in three-protocol
HTTP recorder or retry wrapper. Logical model interaction/usage/error observations
still apply; absent physical `client_requests` must not be described as captured
HTTP attempts. In particular, native decision handoffs such as an unfillable or
uncertain route are not transport failures to retry. Runtime does not add a fallback, replay tools, or bypass approval on its behalf.
Declare Runtime-aware fallback explicitly with `FallbackBinding`, as below.

For images, both the binding's `contract["input_mime_types"]` and the chosen
native adapter must support the format. Recoverable reference inputs use the same
existing authorized hydration lifecycle; a text-only decision adapter does not
gain image support from a binding declaration. Do not infer capabilities or a
protocol from the model's name.

## Explicit fallback bindings

`FallbackBinding` composes existing `ModelBinding` candidates in order. For the
`DecisionBinding` and `Decision` defined above, a language-model fallback can be
registered through the same synchronous route loader:

```python
from pydantic_ai.models.decision import DecisionHandOff
from linktools.ai.model import FallbackBinding, create_model
from linktools.ai.runtime import RuntimeFeatures

decision_binding = DecisionBinding()
language_binding = create_model(protocol="openai-responses", model="gpt-5")

features = RuntimeFeatures("decision-fallback")
features.model("decision-route", lambda: FallbackBinding(
    decision_binding,
    language_binding,
    fallback_on=DecisionHandOff,
))
features.agent("classifier", model="decision-route", allow_tools=())
# Inside Runtime.open(..., features=features):
# result = await runtime.agents.get("classifier").run(prompt, output=Decision)
```

Ordinary custom bindings keep `materialize(self) -> Model` unchanged. Only
`FallbackBinding.materialize(*, prepare=None)` accepts a candidate-preparation
callback; Runtime supplies it internally. One composition supports any number of
ordered candidates. Explicitly nesting a `FallbackBinding` inside another is
rejected at construction; policies are not silently flattened. Applications do
not need another registry, binding protocol, or callback argument on custom
bindings. Each materialization creates a fresh SDK fallback composition; the
caller owns its async context, and the SDK enters and exits candidate models.
Unused candidates can be materialized, but are not requested or input-hydrated.

The SDK interprets `fallback_on`: exception types, synchronous/asynchronous
predicates, annotated response handlers and supported sequences retain their
native meaning on fresh requests. Omitting it uses `(ModelAPIError,)`, which also
handles `DecisionHandOff`. The explicit policy above narrows fallback to handoffs.
Once a suspended response pins a candidate, successful continuation still uses
that candidate, including a non-first candidate. A failed continuation propagates
its original exception after that candidate's own transport retries; it cannot
rewind the fallback chain and reset retry budgets. A continuation rejected by a
native response handler fails rather than accepting the response or replaying
candidates. SDK releases that do not apply response handlers to continuations do
not acquire additional handler calls.

Response handlers are not applied to streamed requests, including Runtime's
streaming observation path. Native pre-first-event behavior varies by SDK release;
Runtime does not add replay after the first event, convert cancellation into a
handoff, or restart tool execution. Request-local continuation state is reset
before yielding a stream and does not affect concurrent fresh requests.

Rejected continuation responses retain available attempt timing and usage. For
the same provider response ID, only a known cumulative increment is billed; a
new ID retains its independent segment usage. If either cumulative observation
omits usage, the increment is unknown (`attempt.usage=None`), not zero. Cost stays
unknown if either endpoint is unpriced; known zero is preserved. Pricing uses the
public response-cost operation where available, not a duplicated price table or
private partial-pricing helper. For these rejected continuations, decreasing measured token or cost totals fail with
`MODEL_RESPONSE_INVALID`; safe details retain numeric before/after observations,
while previously known run usage remains intact. These observations are diagnostic,
not additional charges. Provider-specific details are not assumed monotonic.

A newly rejected suspended job is cancelled only through the candidate actually
called, without interpreting or merging its metadata. The old job remains with
its existing owner: SDK Agent handles its normal failure/cancellation cleanup;
a caller using bare `Model.request` retains responsibility for its old suspended
job. No new durable metadata, model identity or cleanup registry is introduced.

Before an actually selected candidate is called, Runtime checks that candidate's
MIME permissions and input encoding, then loads authorized reference inputs for
that candidate. The composite MIME declaration is a union for initial admission,
not a promise that every candidate accepts every format. Declared input-capability
rejection happens before reference loading or provider IO; hydrated bytes are
checked again before dispatch. Whether an error permits another candidate is still
governed by `fallback_on`. An unused candidate's input policy does not reject an
otherwise valid request. Thinking translation belongs to the selected SDK adapter,
not a composite LinkTools capability table. Reference bytes retain their existing
authorization and omission policy and are not eagerly loaded for unused models.

The route loader still resolves one binding per agent run. Its ordered candidate
identities and non-secret contracts describe the captured configuration; handler
implementations are neither serialized nor hashed. Callers must freeze the
configuration and handler behavior they intend to reuse, and own consistency of
their existing named declarations under the same `id` and `revision`. A callable
closure is not an immutable execution snapshot. `FallbackBinding` introduces no
model revision/identity system and does not change route-based recovery semantics.
It is the explicit composition boundary for candidate-aware Runtime preparation:
do not assume an arbitrary custom binding returning a bare or wrapped SDK
`FallbackModel` receives the same adaptation.

Fallback attempts remain distinct from logical model requests and physical HTTP
`client_requests`. When the SDK exposes failed-attempt facts, Runtime retains their
order, available timing, diagnostics and usage through the existing model-history
and interaction projection, subject to the existing redaction and content-omission
policy. If all candidates fail, available facts are also exposed through the
execution result's `safe_error_details["failed_attempts"]`. Missing facts do not
acquire synthetic attempts, timestamps or zero-valued usage. Reported attempt usage is an
observed subtotal only: missing usage remains unknown, so even a recorded subtotal
is not proof of the complete token usage or cost of a failed fallback chain.
Non-streaming requests retain the SDK's usage accounting. A streamed exhausted
fallback group bypasses that SDK error-recovery path, so Runtime adds its known
attempt-usage subtotal once to run usage and checks the existing usage limits;
the SDK still accounts for any separately produced partial response. History and
interaction projections do not add those amounts again. A custom candidate binding still does not automatically gain the
built-in HTTP recorder or transport retries; absent physical requests are not
evidence that no provider call occurred.

## SDK model settings and native request fields

`create_model(..., model_settings={...})` is the single JSON extension point for
SDK common and provider-specific settings across Chat Completions, Responses and
Anthropic Messages. `max_tokens` and `timeout` live only in this mapping, not in
duplicate factory arguments. LinkTools does not translate proprietary switches;
the selected SDK adapter owns their interpretation. An unknown SDK settings key is not automatically a body field.
Use nested `extra_body` for native endpoint fields:

```python
binding = create_model(
    protocol="anthropic", model="claude-opus-4-5",
    model_settings={
        "max_tokens": 8192,
        "anthropic_thinking": {"type": "enabled", "budget_tokens": 4096},
        "extra_body": {"custom_endpoint_option": True},
    },
)
```

The manual budget applies only to models supporting that control; adaptive-thinking
models may use different settings. The custom body field is illustrative and must
be supported by the endpoint. For raw thinking fields use
`model_settings={"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 4096}}}`.
A native setting and `extra_body` cannot both own the same provider request object.
Conflicts fail before sending, including explicit `None` values: for example,
`anthropic_effort="low"` plus `extra_body.output_config` is ambiguous even if the
latter only supplies `format`. Choose either the SDK-native settings or the whole
raw object, such as `extra_body={"output_config": {"effort": "low", ...}}`, not both.
There is no guessed deep merge. Unknown raw fields and nested values pass through.
The model identity, input messages, registered tools, instructions and streaming
mode belong to the request, and cannot be replaced through `extra_body`.
Native structured-output schemas likewise own their provider output object.

Protocol adapters track the wire-object owners of supported SDK fields, not an
allowlist of extensions. When updating an SDK adapter, update this ownership mapping
and its wire tests together. Unified Thinking is a portable fallback, not a second
native owner; explicit provider settings or a raw thinking object still take
precedence. A compatible backend may ignore numeric thinking budgets, and unified
`False` can mean omission or minimum reasoning rather than a guaranteed wire-level
disable. Use the backend's documented native control when that distinction matters.

Per-request settings replace binding settings by key, following the SDK's shallow
merge: a request-level `extra_body` is the complete replacement object, not a partial
patch. Omitted keys inherit; explicit `None` retains that field's SDK semantics,
not a universal inheritance/reset rule. For example, an explicit `max_tokens=None`
serializes a null limit and may be rejected by the endpoint. Use omission for the
provider default, not null. Raw-only output limits are not interpreted as a local
context reserve; use the standard setting when local budgeting is required.

The binding copies and freezes JSON settings. Its `contract["settings"]` records
the resolved semantic settings, excluding operational `timeout`, `request_timeout`,
`max_retries` and `extra_headers`; timeouts and headers still reach the SDK,
while transport retry ownership remains with LinkTools. Authentication
belongs in connection parameters, not in body fields. Semantic settings and
`extra_body` are contract-captured data, not a secret store. Excluding headers from
the contract does not make arbitrary request logging safe for secrets.

A host can forward its parsed JSON/YAML `model_settings` mapping directly to
`create_model`; configuring these settings does not require writing a custom
`ModelBinding`. This is a Python construction interface, not an automatic new
workspace/CLI configuration loader. Recovery still resolves the current binding
under the registered route, not frozen historical settings.

## Native thinking and migration

Runtime uses the public PydanticAI `Thinking` capability. The Agent spec defaults
to `thinking=None`; an omitted run value inherits the spec. When the resolved value
is `None`, Runtime adds no thinking override, preserving model settings and native
application capabilities. An explicit boolean or effort is appended after application
capabilities as the host choice. Provider-specific settings and final native body
overrides retain SDK precedence over unified thinking.

Values are `None`, `True`, `False`, `"minimal"`, `"low"`, `"medium"`, `"high"`,
`"xhigh"` and `"max"`. `False` is an explicit request for disabled thinking where
possible, or the lowest available effort on always-on models—not inheritance or a
LinkTools validation error. `None` preserves defaults, including across durable
restore and fork. CLI omission inherits; `--thinking` and `--no-thinking` explicitly
select true and false. The Web composer offers an inherit choice as well as explicit
on/off. The SDK translates effort for the selected model; see the official
[PydanticAI Thinking documentation](https://pydantic.dev/docs/ai/capabilities/thinking/).
Details depend on the installed SDK version.

The Web composer no longer disables the thinking selector based on a separate
capability projection. Returned thinking remains displayable independently of the
requested setting. Input MIME authorization and actual protocol encoding remain
separate requirements; neither thinking nor a body extension grants attachment
support.

Migrate the removed pre-release APIs directly; no compatibility aliases remain:

- Move factory `max_tokens=` and `timeout=` into `model_settings`; no aliases remain.
  Keep transport retry parameters on the factory and remove SDK retry-owner overrides.
- Remove `create_model(profile=...)`. Move a declared `context_window` to the
  dedicated argument; its contract location is now top-level `context_window`,
  not `profile.context_window`. Do not move the old profile wholesale into
  `model_settings` or `extra_body`: SDK capability/encoding switches such as
  `openai_chat_supports_document_input` are not wire fields. Their automatic SDK
  inference remains in effect; body extensions cannot enable unsupported SDK
  encoding or teach an adapter a new capability.
- Replace native wire options with `model_settings={"extra_body": {...}}`, using endpoint field names,
  not SDK profile keys such as `supports_thinking`. For ordinary thinking use
  Agent/run `thinking=` instead. SDK capability inference remains automatic;
  SDK common/provider settings go directly in `model_settings`; only custom model
  construction needs a `ModelBinding`. The former top-level `extra_body=` argument
  is removed, not retained as an alias.
- Remove imports/calls of `model_thinking_control` and custom `thinking_control()`
  projections. Do not replace them with a duplicated provider capability table or
  use them to disable controls. Handle actual SDK/provider outcomes instead.

## Output-only requests

The built-in provider bindings default `parallel_tool_calls` to `False` only
after the SDK has resolved a request to tool output with exactly one output
tool, no ordinary function tools, no native tools and no free-text output.
This avoids requesting parallel alternative terminal results; ordinary tool
requests keep their existing parallel-call behavior. The policy follows the
actual request shape, not an Agent name or a separate model route.

Explicit SDK model or request settings, including `True`, take precedence.
An SDK-declared unsupported setting is not added. Native structured/text output
and externally supplied SDK models remain unchanged. This is a request default,
not a new `AgentSpec` or `create_model` option, and it does not change retry or
output-validation ownership. Compatible gateways must actually support and
honor the provider field; no token reduction is inferred from the setting alone.

## Transport retries

The model binding is the sole owner of transport retries; native SDK retries
remain zero for every attempt. `model_settings["max_retries"]` may be omitted or
exactly integer `0`; `None`, nonzero values and booleans are configuration errors,
both at binding construction and for per-request settings. A request override
cannot hide an invalid binding value. SDK zero does **not** disable LinkTools retries:
use `create_model(max_retries=0)` for that. Backoff parameters never silently become
ineffective because of an SDK override.

`max_retries` counts additional attempts (default `2`), and `retry_delay`
is the initial backoff base (default `1.0` seconds), not a fixed wait. The local
base is multiplied by `retry_backoff_factor` (default `2.0`) after each retry
wait, with up to 25% additive jitter. The factor must be finite and at least
`1`: `1` keeps the base unchanged while retaining jitter; larger values grow
the base until capped. `retry_max_delay` (default `60.0` seconds) caps the local
wait, including jitter; the effective
cap is `max(retry_max_delay, retry_delay)`, so it never shortens the initial
base. Both delay parameters accept finite non-negative numbers. Explicit
`retry_delay=0` keeps the local wait at zero; `max_retries=0` disables retries.

For example, `create_model(..., max_retries=3, retry_delay=5,
retry_max_delay=120, retry_backoff_factor=4)` allows three additional attempts
with local waits of 5–6.25, 20–25, and 80–100 seconds. These transport settings
do not change the model's semantic identity or contract.

The SDK's public normalized `retry_after` is a lower bound for the current
response only, even when it exceeds the local cap. A later response without
that hint uses its own exponential base. Caller cancellation and the SDK's
whole-request `request_timeout` interrupt waiting; direct
Model calls without such a caller deadline have no implicit total-wait budget.

Ordinary transient 429/408/409/5xx errors may retry. Known quota exhaustion is
not a temporary rate limit: the SDK's public typed quota error, or its exact
structured `insufficient_quota` code on a model HTTP error, is rejected without
retry. Overall request-deadline exhaustion also does not start a new attempt.
The SDK's typed context-window overflow is also rejected without resending
unchanged input. Typed rate-limit, unavailable, server, phase-timeout and
connection errors retain their categories even without an HTTP status. The
supported SDK minimum provides these public categories. A genuinely opaque
`ModelAPIError` still keeps the bounded generic retry behavior; LinkTools does
not infer a permanent rejection from ambiguous message text.

Malformed Chat completion responses with no choices are rejected at the
provider validation boundary as `MODEL_RESPONSE_INVALID`, not an internal
indexing error. Empty-stream EOF and output validation remain SDK response
semantics rather than blanket transport retries. Valid empty responses can use
the Agent's separate output-correction budget; token-limit exhaustion does not
silently increase `max_tokens` or restart the request.

Before the first event reaches the consumer, an eligible transport failure can
retry within the same policy. Once a partial preview has been exposed, retry
requires a consumer that acknowledges replacement of that failed preview before
new deltas arrive. Runtime's bundled Web and CLI consumers implement this boundary;
a consumer without replacement support does not silently opt in. Native-tool or
suspended continuation state prevents replay, even before the first event, because
it may represent provider-owned work. Cancellation and consumer exit do not start
another attempt.

These transport retries are distinct from Agent output/tool correction retries,
task retries and host rate-limit admission. A logical request hook still runs once
around potentially multiple observed SDK stream invocations and HTTP dispatches;
it is not a physical-attempt quota hook. Their sequence numbers are separate:
native fallback may perform several provider dispatches within one SDK invocation.

## Trace and integrity boundaries

The default model reader uses the existing SDK model interaction projection. It
preserves ordered messages, parts, instructions, tool definitions and results,
compaction boundaries and attachment metadata. It is the prepared SDK input, not a
claim to reproduce the provider's exact HTTP body or hidden chain of thought.
Reasoning summaries, public thinking and opaque signatures/encrypted content remain
distinct. Empty redacted-thinking parts are retained for subsequent native turns.

Raw HTTP is an explicit diagnostic view. `TraceCapability(record_http_bodies=True)`
is opt-in; raw capture remains off by default. Authentication headers are not stored.
Bodies themselves can contain sensitive prompts, tool results, attachments or opaque
provider state, so enable capture only with appropriate access and retention policy.
The existing model-history authorization, storage retention and deletion boundaries
apply; no second session snapshot store is created.

Passive capture observes the same stream consumed by the SDK and preserves native
bytes, attempts and errors. Chat `[DONE]`, Responses completed/failed/incomplete/error,
and Anthropic message_stop/error are protocol terminals. A terminal capture does not
mean a successful model response. Cancellation or disconnection retains only the
observed partial body. Passive capture never initiates retries; replacement follows
the bounded model policy and consumer contract above. Native provider SDK retries
remain disabled.

Passive timing and byte/chunk counts remain available when body capture is off;
that mode does not buffer or parse request/response bodies. Per-attempt
`diagnostics.protocol_terminal` is a terminal name when observed, `null` when
recorded SSE bytes were decoded without a terminal, and absent when unobserved
(including body capture off or decoding failure). `response.complete=False` alone
therefore does not prove a truncated response. A protocol terminal can precede
HTTP EOF because the SDK may stop reading and close the connection first. Inspect
transport `read_outcome` and separate `sdk_*` exhaustion/cancellation/close timing
without treating either as proof of provider success or fault. Observations use
the configured protocol, never inference from the model name. They neither add
model retries nor replay external effects.

PydanticAI 2.55 constructs Anthropic `BetaMessageParam` as a mapping. Anthropic 2.x
changed that symbol to a union, so the dependency is constrained to the supported 1.x
line until the native adapter supports that change. No SDK monkeypatch is applied.
