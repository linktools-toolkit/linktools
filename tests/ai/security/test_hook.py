import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.security.hook import SecurityCapability


def _agent_calling_bash_with(command: str) -> Agent:
    """FunctionModel-driven agent whose first turn always calls `bash` with exactly
    `command`, and whose second turn (reached only if the first tool call's return
    doesn't stop it) just answers with text — verified directly against pydantic-ai
    1.107.0 to actually drive the tool call this way (TestModel's `custom_output_args`
    does NOT do this; it's for structured-output tools, not arbitrary tool args, and
    raises IndexError when there's no output tool defined)."""

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) <= 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="bash", args={"command": command})])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(model_fn), capabilities=[SecurityCapability()])

    @agent.tool_plain
    def bash(command: str) -> dict:
        return {"exit_code": 0, "stdout": f"ran: {command}", "stderr": ""}

    return agent


def _tool_returns(result) -> list:
    return [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if getattr(part, "part_kind", None) == "tool-return"
    ]


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -rf / --no-preserve-root",
        "sudo rm -rf /*",
        "dd if=/dev/zero of=/dev/sda",
        ":(){ :|:& };:",
        "mkfs.ext4 /dev/sda1",
    ],
)
def test_blocks_known_destructive_commands(command):
    agent = _agent_calling_bash_with(command)
    result = agent.run_sync("run it")
    tool_returns = _tool_returns(result)
    assert tool_returns, "expected the bash tool to have been called"
    assert "blocked" in str(tool_returns[0]).lower()
    assert "ran:" not in str(tool_returns[0])


def test_allows_safe_commands():
    agent = _agent_calling_bash_with("ls -la")
    result = agent.run_sync("run it")
    tool_returns = _tool_returns(result)
    assert tool_returns
    assert "ran: ls -la" in str(tool_returns[0])


def test_capability_satisfies_protocol():
    from pydantic_ai.capabilities import AbstractCapability

    assert isinstance(SecurityCapability(), AbstractCapability)
