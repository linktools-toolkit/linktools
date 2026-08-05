"""Register the deterministic model used by the subprocess smoke fixture."""

import os
import runpy
from pathlib import Path

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from linktools.ai.model.registry import model_registry


def _prompt_text(messages):
    for message in reversed(messages):
        for part in getattr(message, "parts", ()):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                return content
    return ""


async def _stream(messages, info):
    prompt = _prompt_text(messages) or repr(messages)
    if "SMOKE_TOOL_" in prompt and not any(
        getattr(part, "part_kind", None) == "tool-return"
        for message in messages
        for part in getattr(message, "parts", ())
    ):
        yield {0: DeltaToolCall(name="bash", json_args='{"command":"printf TOOL_ALLOWED"}', tool_call_id="smoke-tool")}
        return
    if "SMOKE_TOOL_" in prompt:
        yield "TOOL_ALLOWED"
        return
    if "SMOKE_TEXT" in prompt:
        yield "SMOKE_"
        yield "OK"
        return
    yield "SMOKE_"
    yield "OK"


model_registry.register(
    "smoke-fixture",
    model=FunctionModel(stream_function=_stream, model_name="smoke-fixture"),
)

fixture_root = os.environ.get("ACP_SMOKE_FIXTURE")
if fixture_root and Path.cwd().resolve() == Path(__file__).parent.resolve():
    runpy.run_path(str(Path(fixture_root) / "sitecustomize.py"))
