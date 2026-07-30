import asyncio

import pytest

from linktools.ai.agent.assembly import AgentFeatureRef
from linktools.ai.agent.dependencies import AgentDependencies
from linktools.ai.agent.tool.idempotency import operation_id
from linktools.ai.agent.tool.models import (
    ExecuteTool,
    ToolExecutionContext,
)
from linktools.ai.agent.tool.service import ToolExecutionService
from linktools.ai.agent.tool.models import (
    ToolCategory,
    ToolDefinition,
    ToolDescriptor,
    ToolSource,
)
from linktools.ai.agent.tool.models import ToolOperationStatus
from linktools.ai.agent.tool.persistence.memory import LocalToolStateBackend
from linktools.ai.agent.tool.store import ToolStateStore
from linktools.ai.errors import JsonEncodingError, ToolDeniedError
from linktools.ai.execution.context import RunContext
from linktools.ai.execution.domain import RunnableType
from linktools.ai.governance.policy.rule import (
    RiskLevel,
    SideEffectKind,
    ToolContext,
)
from linktools.ai.governance.security.pipeline import (
    PipelineAction,
    PipelineDecision,
)


def _request(
    handler,
    *,
    side_effect: SideEffectKind = SideEffectKind.READ_ONLY,
    call_id: str = "call",
    trace_sink=None,
) -> ExecuteTool:
    run = RunContext(
        "execution",
        "execution",
        None,
        "session",
        "agent",
        RunnableType.AGENT,
        "user",
        "tenant",
        None,
    )
    return ExecuteTool(
        definition=ToolDefinition(
            descriptor=ToolDescriptor(
                name="tool",
                source=ToolSource.EXTENSION,
                category=ToolCategory.EXTENSION_EXECUTE,
                risk=RiskLevel.MEDIUM,
                side_effect=side_effect,
                feature=AgentFeatureRef("extension", "test"),
            ),
            handler=handler,
            handler_revision="handler-v1",
            provider_revision="provider-v1",
        ),
        arguments={"value": 1},
        context=ToolExecutionContext(
            execution_id="execution",
            tool_call_id=call_id,
            dependencies=AgentDependencies(
                tool_context=ToolContext("execution", "session")
            ),
            run_context=run,
            trace_sink=trace_sink,
        ),
    )


@pytest.mark.asyncio
async def test_completed_call_replays_without_reinvoking_handler() -> None:
    calls = 0

    async def handler(value):
        nonlocal calls
        calls += 1
        return {"value": value}

    service = ToolExecutionService(
        state=LocalToolStateBackend()
    )
    class Sink:
        payloads = []

        async def tool_result(self, payload):
            self.payloads.append(payload)

    sink = Sink()
    request = _request(handler, trace_sink=sink)
    assert await service.execute(request) == {"value": 1}
    assert await service.execute(request) == {"value": 1}
    assert calls == 1
    assert [payload["status"] for payload in sink.payloads] == [
        "completed",
        "completed",
    ]
    assert [payload["replayed"] for payload in sink.payloads] == [
        False,
        True,
    ]
    assert all(
        payload["operation_id"] == operation_id("execution", "call")
        for payload in sink.payloads
    )


@pytest.mark.asyncio
async def test_bytes_result_fails_and_records_failure() -> None:
    async def handler(value):
        return b"not-json"

    state = LocalToolStateBackend()
    service = ToolExecutionService(state=state)
    with pytest.raises(JsonEncodingError):
        await service.execute(_request(handler))
    stored = await state.get(operation_id("execution", "call"))
    assert stored is not None
    assert stored.status is ToolOperationStatus.FAILED


@pytest.mark.asyncio
async def test_non_replay_safe_cancellation_is_indeterminate() -> None:
    started = asyncio.Event()

    async def handler(value):
        started.set()
        await asyncio.Event().wait()

    state = LocalToolStateBackend()
    service = ToolExecutionService(state=state)
    task = asyncio.create_task(
        service.execute(
            _request(handler, side_effect=SideEffectKind.DESTRUCTIVE)
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    stored = await state.get(operation_id("execution", "call"))
    assert stored is not None
    assert stored.status is ToolOperationStatus.INDETERMINATE


class DenyingSecurity:
    async def before_tool(self, event):
        return PipelineDecision(PipelineAction.DENY, reason="blocked")


@pytest.mark.asyncio
async def test_security_deny_prevents_handler() -> None:
    called = False

    async def handler(value):
        nonlocal called
        called = True
        return value

    service = ToolExecutionService(
        state=LocalToolStateBackend(),
        security=DenyingSecurity(),
    )
    with pytest.raises(ToolDeniedError, match="blocked"):
        await service.execute(_request(handler))
    assert not called
