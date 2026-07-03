from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.memory.capability import MemoryCapability


def _drive_single_tool_call(tool_name: str, args: dict):
    call_state = {"done": False}

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        if not call_state["done"]:
            call_state["done"] = True
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])
        return ModelResponse(parts=[TextPart("done")])

    return model_fn


def _tool_returns(result):
    return [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if getattr(part, "part_kind", None) == "tool-return"
    ]


def test_read_memory_empty_when_nothing_written(tmp_path):
    cap = MemoryCapability(root=tmp_path)
    agent = Agent(FunctionModel(_drive_single_tool_call("read_memory", {})), capabilities=[cap])
    result = agent.run_sync("read")
    assert _tool_returns(result)[0] == ""


def test_write_then_read_roundtrip(tmp_path):
    cap = MemoryCapability(root=tmp_path)

    agent = Agent(FunctionModel(_drive_single_tool_call("write_memory", {"content": "remember this"})), capabilities=[cap])
    agent.run_sync("write")

    agent2 = Agent(FunctionModel(_drive_single_tool_call("read_memory", {})), capabilities=[MemoryCapability(root=tmp_path)])
    result2 = agent2.run_sync("read")
    assert _tool_returns(result2)[0] == "remember this"


def test_write_persists_to_disk(tmp_path):
    cap = MemoryCapability(root=tmp_path)
    agent = Agent(FunctionModel(_drive_single_tool_call("write_memory", {"content": "hi"})), capabilities=[cap])
    agent.run_sync("write")
    assert (tmp_path / "notes.md").read_text(encoding="utf-8") == "hi"


def test_capability_satisfies_protocol(tmp_path):
    assert isinstance(MemoryCapability(root=tmp_path), AbstractCapability)
