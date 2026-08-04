import pytest

from linktools.ai.execution.live_events import (
    AssistantTextDelta,
    ExecutionAlreadySubscribedError,
    ExecutionCancelled,
    ExecutionEventHub,
    LiveEventConsumerSlowError,
)


@pytest.mark.asyncio
async def test_event_hub_is_execution_scoped_and_terminal() -> None:
    hub = ExecutionEventHub(queue_size=4)
    subscription = await hub.subscribe("one")
    with pytest.raises(ExecutionAlreadySubscribedError):
        await hub.subscribe("one")

    await hub.publish("one", AssistantTextDelta(execution_id="one", text="a"))
    await hub.close("one", ExecutionCancelled(execution_id="one"))

    first = await subscription.__anext__()
    second = await subscription.__anext__()
    assert first.sequence == 1
    assert second.sequence == 2
    assert hub.active_subscription_count == 0


@pytest.mark.asyncio
async def test_event_hub_backpressure_only_fails_target_execution() -> None:
    hub = ExecutionEventHub(queue_size=1, publish_timeout=0.01)
    await hub.subscribe("slow")
    await hub.subscribe("healthy")
    await hub.publish("slow", AssistantTextDelta(execution_id="slow", text="a"))

    with pytest.raises(LiveEventConsumerSlowError):
        await hub.publish("slow", AssistantTextDelta(execution_id="slow", text="b"))

    await hub.publish("healthy", AssistantTextDelta(execution_id="healthy", text="ok"))
