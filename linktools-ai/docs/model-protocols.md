# Model protocols and readable traces

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
features.model("compatible", create_model(protocol="openai", model="gateway-model", profile={
    "supports_thinking": True,
    "thinking_always_enabled": False,
}))

async with Runtime.open("app", features=features, storage=RuntimeStorage.in_memory()) as runtime:
    ...
```

`create_model` returns a `ModelBinding`, not a registry. It accepts input types,
profile, credentials, timeouts and retry/token limits. Missing
provider extras fail with the matching installation hint. Unknown protocol strings
fail with `MODEL_CONFIG_INVALID` without echoing connection values.

The public type values are `openai` (Chat Completions), `openai-responses`, and
`anthropic` (Messages). CLI `--model-api`, `MODEL_API`, workspace `model_api`,
HTTP trace labels and input dispatch use the same values. Native HTTP endpoint
paths are unchanged. There is no separate public registry or `Runtime.open(models=...)`.

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
under that same route; protocol, provider model, profile and connection changes do
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
thinking policy, MIME permissions and input encoding, then loads authorized
reference inputs for that candidate. The composite MIME declaration is a union
for initial admission, not a promise that every candidate accepts every format.
Thinking or declared input-capability rejection happens before reference loading
or provider IO; hydrated bytes are checked again before dispatch. Whether an
error permits another candidate is still governed by `fallback_on`. A valid but
incompatible thinking/MIME policy on an unused candidate does not reject an
otherwise valid request. The composite does not invent a shared thinking profile;
its control projection is unknown. Reference bytes retain their existing
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
execution result's `safe_error_details["failed_attempts"]`. Older SDKs without those
facts do not acquire synthetic attempts, timestamps or zero-valued usage. Reported attempt usage is an
observed subtotal only: missing usage remains unknown, so even a recorded subtotal
is not proof of the complete token usage or cost of a failed fallback chain.
Non-streaming requests retain the SDK's usage accounting. A streamed exhausted
fallback group bypasses that SDK error-recovery path, so Runtime adds its known
attempt-usage subtotal once to run usage and checks the existing usage limits;
the SDK still accounts for any separately produced partial response. History and
interaction projections do not add those amounts again. A custom candidate binding still does not automatically gain the
built-in HTTP recorder or transport retries; absent physical requests are not
evidence that no provider call occurred.

## Explicit profiles

`profile` is a partial mapping of native PydanticAI profile keys to JSON values.
Only explicit declarations are frozen, copied and included in the binding contract.
Omitting it, or passing `{}`, preserves the original Chat contract. It is not a
serialization of the complete inferred SDK profile. Classes, functions and arbitrary
Python objects are rejected; advanced integrations can still register their own
`ModelBinding`. Declared input types and actual protocol encoding must both permit
an attachment. A profile cannot make unsupported wire formats work.

An unknown compatible model name may have no declared thinking control. That does
not mean the server cannot produce reasoning. Declare only capabilities the chosen
SDK adapter can actually encode, such as native OpenAI reasoning effort. Setting
`supports_thinking=True` does not implement a proprietary `enable_thinking` gateway
field. We do not guess vendor-specific switches from model names.

The Web composer reads a small, non-persisted capability projection of the current
agent binding, without resolving credentials or creating a provider client.
Always-on models cannot be switched off. Undeclared control is disabled, not labeled
as proof that the model cannot reason. The model's default and the selected next
request value are distinct. Runtime validation remains authoritative. Returned
`ThinkingPart` content stays visible independently of request control. Custom bindings
without a `thinking_control()` projection report unknown control.

## Transport retries

The model binding owns transport retries; native provider SDK retries remain
zero. `max_retries` counts additional attempts (default `2`), and `retry_delay`
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
whole-request `request_timeout`, when supported, interrupt waiting; direct
Model calls without such a caller deadline have no implicit total-wait budget.

Ordinary transient 429/408/409/5xx errors may retry. Known quota exhaustion is
not a temporary rate limit: the SDK's public typed quota error, or its exact
structured `insufficient_quota` code on a model HTTP error, is rejected without
retry. Overall request-deadline exhaustion also does not start a new attempt.
Supported SDK phase timeout and connection errors retain timeout/unavailable
classification rather than becoming generic API failures. No message-text
heuristics classify quota or infer the configured protocol.

A stream is never replayed once entered, including when the SDK reports an
`in_stream` failure before yielding its response object. These transport retries
are distinct from Agent output/tool correction retries, task retries and host
rate-limit admission. A logical request hook still runs once around potentially
multiple HTTP attempts; it is not a physical-attempt quota hook.

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
observed partial body; consumed streams are not replayed automatically. SDK retry is
disabled in favor of the existing bounded model retry wrapper.

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
