#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deferred approvals retain the actual native model-step capability identity."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic_ai import Agent, DeferredToolRequests, Tool
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolResults
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.step_persistence import StepPersistence
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import AssetCapability, CapabilityContribution, ExecutionCapability, tool_metadata
from linktools.ai.core import ApprovalDecision, ExecutionStatus, Principal
from linktools.ai.errors import ErrorCode
from linktools.ai.execution import ApprovalDecisionRequest, PendingDeferredCall
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceStorage, SqlStepStore, codec
from linktools.ai.persistence.runs import StagingAgentRunStore
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._mcp import _model_tool_name
from linktools.ai.runtime._run_hooks import AgentRunCapture
from linktools.ai.runtime._tool_boundary import TOOL_CAPABILITY_REF_METADATA_KEY
from linktools.ai.spec import CapabilityRef, MCPServerSpec
from linktools.ai.storage import StoredPayload

from ._runtime_test_helpers import _Models, _UsageFunctionModel, _wait_for_committed, model_features
from .test_current_feature_resolution import _store


@asynccontextmanager
async def _runtime(database: Path, store, declarations, model, *, validation_shortcut: bool = False):
    class Loader:
        async def load(self, context):
            return tuple(declarations)

    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    await provision_database(engine)
    steps = SqlStepStore(engine, namespace="deferred-identity", tenant_id="default")
    await steps.initialize()
    features = RuntimeFeatures("deferred-identity")
    assets = AssetCapability(store)
    assets.loader("test", Loader())
    features.capability(assets)
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(StepPersistence(store=steps, capture_frontier=True))

    class ReturnedMetadata(AbstractCapability):
        id = "test.returned-metadata"

        async def after_run(self, ctx, *, result):
            if isinstance(result.output, DeferredToolRequests):
                for call in result.output.approvals:
                    result.output.metadata[call.tool_call_id] = {
                        TOOL_CAPABILITY_REF_METADATA_KEY: CapabilityRef("tool", "checked", 99).to_payload(),
                    }
            return result

    features.capability(ReturnedMetadata())
    if validation_shortcut:
        class CachedValidation(AbstractCapability):
            id = "test.cached-validation"

            def get_ordering(self):
                return CapabilityOrdering(position="outermost")

            async def wrap_tool_validate(self, ctx, *, call, tool_def, args, handler):
                return {}

            async def before_tool_validate(self, ctx, *, call, tool_def, args):
                pytest.fail("cached validation must skip the inner validation lifecycle")

        features.capability(CachedValidation())
    features.agent("default", tool_retries=0)
    model_features(_Models(_UsageFunctionModel(model)), features)
    try:
        async with Persistence.open("deferred-identity", storage=PersistenceStorage.sql(engine)) as persistence:
            async with Runtime.attach(persistence=persistence, features=features.borrow(), auto_recover=False) as runtime:
                yield runtime, persistence
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "resumed_revision", "validation_shortcut"), (
    ("tool", 1, False), ("tool", 2, False), ("tool", None, False),
    ("mcp", 1, False), ("mcp", 2, False), ("mcp", None, False),
    ("tool", 1, True), ("tool", 2, True),
))
async def test_native_approval_preserves_provided_ref_across_source_change_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, resumed_revision: int | None,
    validation_shortcut: bool,
) -> None:
    effects = []
    declarations = []
    tool_name = "checked" if kind == "tool" else _model_tool_name("server", "checked")

    def candidate(value: str, revision: int):
        if kind == "mcp":
            return MCPServerSpec("server", transport="streamable-http", url=f"https://{value}.invalid/mcp",
                                 revision=revision)

        def checked() -> str:
            effects.append(value)
            return value

        return CapabilityContribution.from_opaque("tool", "checked", Tool(
            checked, requires_approval=True,
            metadata=tool_metadata(effect_policy="replay_safe" if value == "original" else "none", tool_class="business"),
        ), revision=revision)

    if kind == "mcp":
        import linktools.ai.runtime._mcp as module

        class FakeMCP(FunctionToolset):
            def __init__(self, transport, *, server):
                def checked() -> str:
                    effects.append(server.url)
                    return server.url
                super().__init__([Tool(checked, requires_approval=True)], id=f"mcp:{server.id}")

            async def close_resources(self) -> None:
                pass

        monkeypatch.setattr(module, "MCPToolset", FakeMCP)

    declarations[:] = [candidate("original", 1)]

    async def model(messages, info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("done")])
        assert any(tool.name == tool_name for tool in info.function_tools)
        # The source changes before observation. The issued model-step definition is still v1.
        declarations[:] = [] if resumed_revision is None else [candidate("replacement", resumed_revision)]
        return ModelResponse(parts=[ToolCallPart(tool_name, {}, tool_call_id="approval-call")])

    store = await _store()
    database = tmp_path / "approval.db"
    try:
        async with _runtime(database, store, declarations, model,
                            validation_shortcut=validation_shortcut) as (runtime, persistence):
            execution = await runtime.agents.get().start("check", idempotency_key="start")
            await _wait_for_committed(
                lambda: persistence.execution.executions.get(execution.execution_id, tenant_id="default"),
                lambda value: value is not None and value.status is ExecutionStatus.WAITING_DEFERRED,
            )
            await asyncio.wait_for(runtime.handoff(), 5)
            waiting = await persistence.execution.executions.get_recovery(execution.execution_id, tenant_id="default")
            pending, = waiting.recovery.pending_tools.approvals
            expected = CapabilityRef(kind, "checked" if kind == "tool" else "server", 1)
            assert pending.capability_ref == expected
            assert pending.metadata[TOOL_CAPABILITY_REF_METADATA_KEY]["revision"] == 99
            assert effects == []
            assert await persistence.checkpoint.tools.list_by_agent_run(
                waiting.recovery.agent_run_id, tenant_id="default", execution_id=execution.execution_id,
            ) == ()

        async with _runtime(database, store, declarations, model,
                            validation_shortcut=validation_shortcut) as (runtime, persistence):
            recovered = await persistence.execution.executions.get_recovery(execution.execution_id, tenant_id="default")
            assert recovered.recovery.pending_tools.approvals[0].capability_ref == expected
            principal = Principal("approver", "default", "service")
            approval, = await runtime.approvals.list(execution.execution_id, principal=principal)
            await runtime.approvals.decide(execution.execution_id, ApprovalDecisionRequest(
                principal, approval.approval_id, "approve", ApprovalDecision.APPROVE,
                metadata={TOOL_CAPABILITY_REF_METADATA_KEY: CapabilityRef(kind, expected.id, resumed_revision or 1).to_payload()},
            ))
            outcome = await runtime.executions.wait(execution.execution_id, principal=runtime.default_principal,
                                                    timeout_seconds=5)
            if resumed_revision == 1:
                assert outcome.result.status is ExecutionStatus.SUCCEEDED, outcome.result
                assert effects == ["replacement" if kind == "tool" else "https://replacement.invalid/mcp"]
            else:
                assert outcome.result.status is ExecutionStatus.FAILED, outcome.result
                if resumed_revision is not None:
                    assert outcome.result.error_code == ErrorCode.CAPABILITY_RESOLUTION_INVALID.value
                assert effects == []
    finally:
        await store.close()


def _capture(run_id: str, references) -> AgentRunCapture:
    return AgentRunCapture(
        recorder=AgentRunRecorder(StagingAgentRunStore(), execution_id=None, agent_run_id=run_id),
        agent_id="generic", agent_run_id=run_id, deferred_pause_sink=lambda _: None,
        deferred_capability_refs=references,
    )


@pytest.mark.asyncio
async def test_generic_native_approval_retains_sdk_semantics_without_invented_identity() -> None:
    effects, references = [], {}

    def tool(value: str) -> Tool:
        def checked() -> str:
            effects.append(value)
            return value
        return Tool(checked, requires_approval=True)

    agent = Agent(TestModel(call_tools=["checked"], custom_output_text="done"), tools=[tool("original")],
                  output_type=[str, DeferredToolRequests])
    result = await agent.run("check", run_id="original", capabilities=[_capture("original", references)])
    assert isinstance(result.output, DeferredToolRequests)
    call, = result.output.approvals
    assert references == {} and effects == []
    pending = PendingDeferredCall(call.tool_call_id, call.tool_name, StoredPayload.inline_json({}), operation_id="test-operation")
    assert pending.capability_ref is None
    agent = Agent(TestModel(call_tools=[], custom_output_text="done"), tools=[tool("replacement")],
                  output_type=[str, DeferredToolRequests])
    resumed = await agent.run(message_history=result.all_messages(), run_id="resumed",
                             deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: True}),
                             capabilities=[_capture("resumed", {})])
    assert resumed.output == "done" and effects == ["replacement"]


@pytest.mark.parametrize("persisted", (False, True))
def test_pending_capability_reference_round_trips_current_wire(persisted: bool) -> None:
    pending = PendingDeferredCall("call", "checked", StoredPayload.inline_json({}),
                                  capability_ref=CapabilityRef("tool", "checked", 1), operation_id="test-operation")
    encoded = codec._encode_domain(pending, codec._CURRENT_CODEC, persisted=persisted)
    assert codec._decode_domain(encoded, PendingDeferredCall, codec._CURRENT_CODEC, persisted=persisted) == pending
