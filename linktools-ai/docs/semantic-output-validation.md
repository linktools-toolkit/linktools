# Semantic output validation

An output schema validates structure. A registered capability can additionally
check whether a structurally valid answer refers to the original execution's
admitted input, without putting every allowed identity into a schema enum.

Use the existing `after_output_validate` hook and `ModelRetry`:

```python
from typing import Any

from pydantic_ai import ModelRetry
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext
from linktools.ai.capability import AgentContext


class EvidenceMembership(AbstractCapability[AgentContext[None]]):
    id = "evidence-membership"

    async def after_output_validate(
        self, ctx: RunContext[AgentContext[None]], *, output_context: Any, output: Any,
    ) -> Any:
        original = ctx.deps.original_input
        if original is None:
            raise RuntimeError("Original input is unavailable")
        allowed = {
            item["input_identifier"]
            for item in original["attachments"]
            if item["input_identifier"] is not None
        }
        selected = output["sources"]
        if len(set(selected)) != len(selected) or not set(selected).issubset(allowed):
            raise ModelRetry("Use distinct source identities from the admitted input.")
        return output


features.capability(EvidenceMembership(), revision=1, scope="agent")
features.agent(
    "reviewer",
    output_schema={
        "type": "object",
        "properties": {"sources": {"type": "array", "items": {"type": "string"}}},
        "required": ["sources"],
        "additionalProperties": False,
    },
    output_retries=1,
)
```

This example assumes the host assigned unique, nonempty input identifiers at
admission. When identifiers are not unique, use the original attachment
occurrence identity or another unambiguous key from the accepted descriptor.
The validator should match the business meaning of the output, including any
required correspondence between an identifier and its exact Asset version.

The membership set comes from `AgentContext.original_input`, the existing
stored-input owner's detached read-only view. It is not reconstructed from
current session state, a newly supplied list, or a process-local closure. For
other captured context, use `AgentContext.input_context`. Native session
`session_metadata` can still change independently. See
[Original execution input](original-execution-input.md).

The registered capability's explicit id and revision are retained in the
Agent execution binding. A reopened Runtime must register that same named
behavior. The host is responsible for implementing a given revision
consistently; Python callbacks are not serialized or recovered automatically.

A reference or input identifier is not an authorization grant. The host must
admit the input for the original principal and keep controlled loading tied to
that admission, its exact descriptor, and any applicable revocation policy.
The validator checks membership in that original set; it does not grant file
access or permission to execute tools. Historical tool output is not initial
input merely because it appears in conversation history.

`ModelRetry` returns correction feedback through the existing output retry
mechanism. Do not add a second correction loop or restart a successful tool to
correct an answer. The output schema contains only the compact structural
contract. Neither the schema nor correction feedback needs the full allowed
set. The original material itself is still delivered according to the normal
input and model-context policies.

Same-version continuation uses `await runtime.handoff()` before reopening the
same persistence. Ordinary Runtime close cancels unfinished work. A passing
membership check also does not make a new, separately requested tool call
idempotent: normal tool effect and replay policies still apply.
