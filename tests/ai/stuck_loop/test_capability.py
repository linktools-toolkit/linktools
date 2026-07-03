from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.stuck_loop.capability import StuckLoopCapability


def test_short_circuits_after_max_repeats():
    calls = {"n": 0}
    turn = {"n": 0}

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        turn["n"] += 1
        # Keep calling the same failing tool with the same args for up to 6 turns;
        # if StuckLoopCapability works, the real tool only ever runs 3 times (the
        # 4th+ attempts get short-circuited before reaching the tool function), and
        # the model "gives up" once it's made 6 attempts either way.
        if turn["n"] <= 6:
            return ModelResponse(parts=[ToolCallPart(tool_name="flaky", args={"x": 1})])
        return ModelResponse(parts=[TextPart("giving up")])

    agent = Agent(FunctionModel(model_fn), capabilities=[StuckLoopCapability(max_repeats=3)])

    @agent.tool_plain
    def flaky(x: int) -> dict:
        calls["n"] += 1
        return {"error": "always fails"}

    agent.run_sync("call flaky")
    assert calls["n"] == 3, f"expected the real tool to run exactly 3 times, ran {calls['n']}"


def test_different_args_are_not_conflated():
    calls = {"n": 0}
    # x=1 and x=2 each fail twice -- with a 2-repeat threshold tuned to a single
    # signature repeating, neither should get short-circuited, since neither
    # individual signature (x=1 or x=2) repeats more than twice in a row on its own.
    sequence = [{"x": 1}, {"x": 2}, {"x": 1}, {"x": 2}, "stop"]
    turn = {"n": 0}

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        item = sequence[turn["n"]] if turn["n"] < len(sequence) else "stop"
        turn["n"] += 1
        if item == "stop":
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart(tool_name="flaky2", args=item)])

    agent = Agent(FunctionModel(model_fn), capabilities=[StuckLoopCapability(max_repeats=2)])

    @agent.tool_plain
    def flaky2(x: int) -> dict:
        calls["n"] += 1
        return {"error": "fails"}

    agent.run_sync("call flaky2")
    assert calls["n"] == 4, f"expected all 4 attempts to reach the real tool, got {calls['n']}"


def test_capability_satisfies_protocol():
    assert isinstance(StuckLoopCapability(), AbstractCapability)
