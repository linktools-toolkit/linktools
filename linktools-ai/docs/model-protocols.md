# Model protocols and readable traces

Install only the provider adapters you use:

```shell
pip install "linktools-ai[openai]"       # Chat Completions and Responses
pip install "linktools-ai[anthropic]"    # Anthropic Messages
pip install "linktools-ai[openai,anthropic,web]"  # both adapters and Web console
```

The base package does not require either provider SDK. Core ModelBinding, Runtime
and Web modules remain importable without them (serving Web additionally needs the
`web` extra). Calling a provider factory/register without its extra fails immediately
with the matching install hint, before connection settings or credentials are resolved.

Choose the actual API, not a vendor-looking model name:

```python
from linktools.ai.model import ModelRegistry

models = ModelRegistry.openai_responses(model="gpt-5")
models.register_anthropic("messages", model="claude-sonnet-4-5")
models.register_openai("compatible", model="gateway-model", profile={
    "supports_thinking": True,
    "thinking_always_enabled": False,
})
```

The public model types and Python entry points are:

| Type | Default-route factory | Route registration |
| --- | --- | --- |
| `openai` | `ModelRegistry.openai(...)` | `models.register_openai(route_id, ...)` |
| `openai-responses` | `ModelRegistry.openai_responses(...)` | `models.register_openai_responses(route_id, ...)` |
| `anthropic` | `ModelRegistry.anthropic(...)` | `models.register_anthropic(route_id, ...)` |

CLI `--model-api`, `AI_MODEL_API`, workspace `model_api`, HTTP trace API labels,
and protocol dispatch use these same type values. Python methods use underscores
where a type contains a hyphen. Native HTTP endpoint paths are unchanged.
The earlier pre-release `openai-chat` / `anthropic-messages` type spellings are not
aliases; update configuration to the current spellings. A captured pre-release
Anthropic contract with the previous discriminator is not a valid restore target
for the current binding; do not edit stored snapshots to bypass contract checks.
Existing Python registration names do not change. Chat's binding contract continues to omit the API field, preserving
its existing contract/digest; readers interpret that established omission as
`openai`, while Responses and Anthropic carry an explicit API discriminator.

The three factory/register pairs have symmetric connection and input-type options.
OpenAI Chat is unchanged; new binding contracts distinguish `openai-responses`
and `anthropic`. Provider/model identity remains separate from the API.
An API switch, profile change, or different captured binding is not an interchangeable
restore target, even if the displayed model name matches.

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

PydanticAI 2.55 constructs Anthropic `BetaMessageParam` as a mapping. Anthropic 2.x
changed that symbol to a union, so the dependency is constrained to the supported 1.x
line until the native adapter supports that change. No SDK monkeypatch is applied.
