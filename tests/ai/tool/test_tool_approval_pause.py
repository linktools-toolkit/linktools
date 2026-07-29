from dataclasses import replace

import pytest

from linktools.ai.agent.assembly import AgentFeatureRef
from linktools.ai.agent.dependencies import AgentDependencies
from linktools.ai.agent.tool.execution.models import (
    ExecuteTool,
    ToolExecutionContext,
)
from linktools.ai.agent.tool.execution.service import ToolExecutionService
from linktools.ai.agent.tool.models import (
    ToolCategory,
    ToolDefinition,
    ToolDescriptor,
    ToolSource,
)
from linktools.ai.agent.tool.policy.resolver import ResolvedToolPolicy
from linktools.ai.agent.tool.state.persistence.memory import LocalToolStateBackend
from linktools.ai.agent.tool.state.store import ToolStateStore
from linktools.ai.errors import RunPaused, ToolIdempotencyConflictError
from linktools.ai.execution.context import RunContext
from linktools.ai.execution.domain import RunnableType
from linktools.ai.governance.policy.rule import (
    RiskLevel,
    SideEffectKind,
    ToolContext,
)


class ApprovalPolicy:
    async def resolve(self, descriptor, context):
        return ResolvedToolPolicy(require_approval=True)


async def _handler() -> str:
    raise AssertionError("approval must happen before the handler")


@pytest.mark.asyncio
async def test_approval_policy_pauses_before_handler() -> None:
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
    definition = ToolDefinition(
        descriptor=ToolDescriptor(
            name="write",
            source=ToolSource.EXTENSION,
            category=ToolCategory.EXTENSION_EXECUTE,
            risk=RiskLevel.HIGH,
            side_effect=SideEffectKind.NAMESPACE_MUTATING,
            feature=AgentFeatureRef("extension", "test"),
        ),
        handler=_handler,
    )
    request = ExecuteTool(
        definition=definition,
        arguments={},
        context=ToolExecutionContext(
            execution_id="execution",
            tool_call_id="call",
            dependencies=AgentDependencies(
                tool_context=ToolContext("execution", "session")
            ),
            run_context=run,
        ),
    )
    with pytest.raises(RunPaused) as caught:
        await ToolExecutionService(policy=ApprovalPolicy()).execute(request)
    assert caught.value.tool_call_id == "call"
    assert caught.value.binding["fingerprint"]


@pytest.mark.asyncio
async def test_approved_binding_executes_once_without_repausing() -> None:
    calls = 0

    async def handler():
        nonlocal calls
        calls += 1
        return "done"

    definition = ToolDefinition(
        descriptor=ToolDescriptor(
            name="write",
            source=ToolSource.EXTENSION,
            category=ToolCategory.EXTENSION_EXECUTE,
            risk=RiskLevel.HIGH,
            side_effect=SideEffectKind.NAMESPACE_MUTATING,
            feature=AgentFeatureRef("extension", "test"),
        ),
        handler=handler,
        handler_revision="handler-v1",
    )
    request = ExecuteTool(
        definition=definition,
        arguments={},
        context=ToolExecutionContext(
            execution_id="execution",
            tool_call_id="call",
            dependencies=AgentDependencies(
                tool_context=ToolContext("execution", "session")
            ),
            run_context=RunContext(
                "execution",
                "execution",
                None,
                "session",
                "agent",
                RunnableType.AGENT,
                "user",
                "tenant",
                None,
            ),
        ),
    )
    service = ToolExecutionService(
        state=LocalToolStateBackend(),
        policy=ApprovalPolicy(),
    )
    with pytest.raises(RunPaused) as caught:
        await service.execute(request)
    approved = replace(
        request,
        context=replace(
            request.context,
            approved_tool_call_id="call",
            approved_binding_fingerprint=caught.value.binding["fingerprint"],
        ),
    )
    assert await service.execute(approved) == "done"
    assert calls == 1

    drifted = replace(
        approved,
        definition=replace(definition, handler_revision="handler-v2"),
    )
    with pytest.raises(ToolIdempotencyConflictError):
        await service.execute(drifted)
