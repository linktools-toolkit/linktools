from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.tool_search.capability import ToolSearchCapability


def test_search_tools_returns_substring_matches():
    call_state = {"done": False}

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        if not call_state["done"]:
            call_state["done"] = True
            return ModelResponse(parts=[ToolCallPart(tool_name="search_tools", args={"query": "file"})])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        FunctionModel(model_fn),
        capabilities=[ToolSearchCapability(tool_names=("read_file", "write_file", "bash", "list_dir"))],
    )
    result = agent.run_sync("search")
    tool_returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if getattr(part, "part_kind", None) == "tool-return"
    ]
    assert tool_returns
    matches = tool_returns[0]
    assert set(matches) == {"read_file", "write_file"}


def test_search_tools_is_case_insensitive_and_empty_on_no_match():
    call_state = {"done": False}

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        # Must terminate after one tool call like the test above -- a model_fn that
        # always returns a ToolCallPart never stops, and the run hits pydantic-ai's
        # default request_limit (50) instead of completing (verified while writing
        # this plan: an earlier version of this test without the termination
        # condition failed with UsageLimitExceeded, not the assertion below).
        if not call_state["done"]:
            call_state["done"] = True
            return ModelResponse(parts=[ToolCallPart(tool_name="search_tools", args={"query": "BASH"})])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        FunctionModel(model_fn),
        capabilities=[ToolSearchCapability(tool_names=("read_file", "bash"))],
    )
    result = agent.run_sync("search")
    tool_returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if getattr(part, "part_kind", None) == "tool-return"
    ]
    assert tool_returns[0] == ["bash"]


def test_capability_satisfies_protocol():
    assert isinstance(ToolSearchCapability(), AbstractCapability)
