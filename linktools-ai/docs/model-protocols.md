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
profile, connection resolver, credentials, timeouts and retry/token limits. Missing
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
base doubles per retry, with up to 25% additive jitter, capped at 60 seconds or
the configured initial base if larger. Explicit zero retains immediate retries;
`max_retries=0` disables them.

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
