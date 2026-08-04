from linktools.ai.acp.event_mapper import AcpEventMapper
from linktools.ai.execution.live_events import (
    AssistantTextDelta,
    ToolCallCompleted,
)


def test_event_mapper_uses_real_execution_and_tool_ids() -> None:
    mapper = AcpEventMapper()
    message = mapper.map(
        AssistantTextDelta(execution_id="execution", text="hello")
    )
    tool = mapper.map(
        ToolCallCompleted(
            execution_id="execution",
            tool_call_id="tool-7",
            tool_name="read_file",
            arguments={"path": "a.txt"},
            result="ok",
        )
    )

    assert message.session_update == "agent_message_chunk"
    assert tool.tool_call_id == "tool-7"
    assert tool.raw_input == {"path": "a.txt"}
    assert tool.raw_output == "ok"
