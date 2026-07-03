from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, SystemPromptPart, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.periodic_reminder.capability import PeriodicReminderCapability


def test_injects_reminder_once_threshold_crossed():
    captured = []

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        captured.append(list(messages))
        return ModelResponse(parts=[TextPart("ok")])

    # threshold_ratio=0.5, max_messages=2 -> threshold crosses once there are >= 1 message.
    agent = Agent(FunctionModel(model_fn), capabilities=[PeriodicReminderCapability(max_messages=2, threshold_ratio=0.5)])
    agent.run_sync("hello")

    assert captured, "model should have been called"
    last_call_messages = captured[-1]
    reminder_texts = [
        part.content
        for message in last_call_messages
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]
    assert any("context" in text.lower() or "reminder" in text.lower() for text in reminder_texts), (
        f"expected a reminder SystemPromptPart, got parts: {[type(p).__name__ for m in last_call_messages for p in m.parts]}"
    )


def test_no_reminder_below_threshold():
    captured = []

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        captured.append(list(messages))
        return ModelResponse(parts=[TextPart("ok")])

    # threshold_ratio=0.9, max_messages=100 -> a single user message is nowhere near threshold.
    agent = Agent(FunctionModel(model_fn), capabilities=[PeriodicReminderCapability(max_messages=100, threshold_ratio=0.9)])
    agent.run_sync("hello")

    last_call_messages = captured[-1]
    reminder_texts = [
        part.content
        for message in last_call_messages
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]
    assert not reminder_texts


def test_capability_satisfies_protocol():
    assert isinstance(PeriodicReminderCapability(), AbstractCapability)
