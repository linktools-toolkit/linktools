from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.plan.capability import PlanCapability
from linktools.ai.resource.local import LocalAgentArtifactStore


def _drive_single_tool_call(tool_name: str, args: dict):
    """FunctionModel that calls one tool once then stops -- the pattern established
    (and required, per a known pydantic-ai request_limit pitfall) in the prior plan's
    tool_search tests."""
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


def test_write_then_list_todos(tmp_path):
    store = LocalAgentArtifactStore(root=tmp_path)
    cap = PlanCapability(session_id="s1", artifact_store=store)

    todos = [{"id": "1", "content": "do the thing", "status": "pending"}]
    agent = Agent(FunctionModel(_drive_single_tool_call("write_todos", {"todos": todos})), capabilities=[cap])
    result = agent.run_sync("write todos")
    assert _tool_returns(result)[0]["todos"] == todos

    agent2 = Agent(FunctionModel(_drive_single_tool_call("list_todos", {})), capabilities=[cap])
    result2 = agent2.run_sync("list todos")
    assert _tool_returns(result2)[0] == todos


def test_update_todo_changes_status(tmp_path):
    store = LocalAgentArtifactStore(root=tmp_path)
    cap = PlanCapability(session_id="s1", artifact_store=store)

    todos = [{"id": "1", "content": "do the thing", "status": "pending"}]
    agent = Agent(FunctionModel(_drive_single_tool_call("write_todos", {"todos": todos})), capabilities=[cap])
    agent.run_sync("write todos")

    agent2 = Agent(FunctionModel(_drive_single_tool_call("update_todo", {"todo_id": "1", "status": "done"})), capabilities=[cap])
    agent2.run_sync("update")

    agent3 = Agent(FunctionModel(_drive_single_tool_call("list_todos", {})), capabilities=[cap])
    result3 = agent3.run_sync("list")
    assert _tool_returns(result3)[0] == [{"id": "1", "content": "do the thing", "status": "done"}]


def test_list_todos_empty_when_nothing_written(tmp_path):
    store = LocalAgentArtifactStore(root=tmp_path)
    cap = PlanCapability(session_id="s1", artifact_store=store)
    agent = Agent(FunctionModel(_drive_single_tool_call("list_todos", {})), capabilities=[cap])
    result = agent.run_sync("list")
    assert _tool_returns(result)[0] == []


def test_capability_satisfies_protocol():
    assert isinstance(PlanCapability(session_id="s1", artifact_store=None), AbstractCapability)
