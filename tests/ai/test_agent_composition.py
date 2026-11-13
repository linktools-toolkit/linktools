#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression coverage for the final Agent composition contract."""
from linktools.ai.model import create_model

from contextlib import asynccontextmanager
from dataclasses import dataclass
import pickle
from types import SimpleNamespace

import pytest
from linktools import ai
from linktools.ai.agent import AgentBindingContract, AgentCompiler
from linktools.ai.capability import CapabilityContribution, SkillDefinition
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.errors import AIError, ErrorCode

from linktools.ai.execution import ExecutionHandle, ExecutionRequest, ResumeSessionRequest
from linktools.ai.runtime._session import RuntimeSessionService
from linktools.ai.spec import CapabilityRef, AgentSpec, MCPServerSpec, SkillSpec
from linktools.ai.core import Principal, PrincipalKind, SessionStatus
from pydantic_ai.capabilities import AbstractCapability


def test_top_level_exports_preserve_identity_and_pickle() -> None:
    from linktools.ai import capability, execution, runtime, workspace

    expected = {
        "Agent": runtime.Agent,
        "RuntimeFeatures": runtime.RuntimeFeatures,
        "Execution": execution.Execution,
        "AgentContext": capability.AgentContext,
        "Runtime": runtime.Runtime,
        "Session": runtime.Session,
        "Workspace": workspace.Workspace,
    }
    assert set(expected) <= set(dir(ai))
    for name, value in expected.items():
        assert getattr(ai, name) is value
        assert pickle.loads(pickle.dumps(getattr(ai, name))) is value
    imported: dict[str, object] = {}
    exec("from linktools.ai import *", imported)
    assert {name: imported[name] for name in expected} == expected


@pytest.mark.parametrize("first", ["capability", "agent", "runtime"])
def test_runtime_feature_annotations_resolve_in_any_import_order(first: str) -> None:
    import subprocess
    import sys

    code = f"""
import importlib
from typing import get_type_hints
importlib.import_module('linktools.ai.{first}')
from linktools.ai import RuntimeFeatures as RootFeatures
from linktools.ai.runtime import RuntimeFeatures, RuntimeFeaturesResolution, RuntimeFeaturesSource
from linktools.ai.capability import CapabilityContribution, capture_contributions
assert RootFeatures is RuntimeFeatures
assert get_type_hints(RuntimeFeatures.resolve)['return'].__origin__ is RuntimeFeaturesResolution
assert get_type_hints(RuntimeFeatures.borrow)['return'].__origin__ is RuntimeFeaturesSource
assert get_type_hints(RuntimeFeaturesSource.resolve)['return'].__origin__ is RuntimeFeaturesResolution
assert get_type_hints(RuntimeFeaturesResolution)['contributions'].__args__[0].__origin__ is CapabilityContribution
for owner in (RuntimeFeatures, RuntimeFeaturesResolution, RuntimeFeaturesSource):
    for name, value in vars(owner).items():
        if isinstance(value, property):
            value = value.fget
        if callable(value) and (not name.startswith('_') or name == '__init__'):
            assert get_type_hints(value)
assert get_type_hints(capture_contributions)['return'].__origin__ is tuple
assert get_type_hints(CapabilityContribution.validate_revision)['value'] is int
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_agent_binding_contract_persists_binding_inputs() -> None:
    binding_contract = AgentBindingContract(
        agent_spec=AgentSpec("agent", model="model", tool_retries=3, output_retries=3),
        model_contract={"version": 1, "id": "model"},
        selected=(),
        subagents=(),
        output_mode="structured",
        output_schema={"type": "object", "properties": {"value": {"type": "string"}}},
    )

    payload = binding_contract.to_payload()

    assert set(payload) == {
        "version",
        "agent_spec",
        "model_contract",
        "selected",
        "subagents",
        "output_mode",
        "output_schema",
    }
    assert "binding_digest" not in payload
    assert len(binding_contract.binding_digest) == 64


@pytest.mark.parametrize("kind", ("agent", "tool", "skill", "mcp", "capability"))
def test_capability_reference_persists_only_named_identity(kind: str) -> None:
    reference = CapabilityRef(kind, "guardrail", 3)
    payload = reference.to_payload()

    assert payload == {"kind": kind, "id": "guardrail", "revision": 3}
    assert CapabilityRef.from_payload(payload) == reference

    decoded = CapabilityRef.from_payload({**payload, "future": True})
    assert decoded == reference
    assert "future" not in decoded.to_payload()


def test_agent_binding_contract_preserves_unknown_fields() -> None:
    payload = AgentBindingContract(
        agent_spec=AgentSpec("agent", model="model", tool_retries=3, output_retries=3),
        model_contract={"version": 1, "id": "model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "object"},
    ).to_payload()
    payload["future"] = 3

    decoded = AgentBindingContract.from_payload(payload)

    assert decoded.to_payload()["future"] == 3


@pytest.mark.parametrize("mode", ([], {}))
def test_binding_decoder_rejects_non_scalar_output_modes(mode: object) -> None:
    payload = AgentBindingContract(
        agent_spec=AgentSpec("agent", tool_retries=3, output_retries=3),
        model_contract={"model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "object"},
    ).to_payload()
    payload["output_mode"] = mode
    with pytest.raises(AIError) as error:
        AgentBindingContract.from_payload(payload)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@dataclass
class _OrderedCapability(AbstractCapability[object]):
    id: str


@pytest.mark.asyncio
async def test_binding_preserves_declaration_order_and_capability_registration_order() -> None:
    group = RuntimeFeatures("application")
    group.capability(_OrderedCapability("z-capability"), revision=2, scope="agent")
    group.capability(_OrderedCapability("a-capability"), revision=3, scope="agent")

    def business(context: object) -> str:
        del context
        return "ready"

    group.tool(business, name="z-tool", revision=4)
    group.tool(business, name="a-tool", revision=5)
    group.mcp(MCPServerSpec("server", "python", revision=6))
    capture = await group.resolve()
    skill = CapabilityContribution.from_declaration(
        SkillDefinition(SkillSpec("guide", "instructions", revision=7))
    )
    spec = AgentSpec("agent")
    compiler = AgentCompiler(
        models={"default": create_model(api='openai', model="test-model")},
        candidates=(*capture.contributions, skill),
        agents={spec.id: spec},
    )
    binding = compiler.bind(compiler.compile(spec))
    refs = binding.binding_contract.selected
    assert refs == (
        CapabilityRef("capability", "z-capability", 2),
        CapabilityRef("capability", "a-capability", 3),
    )
    assert binding.binding_contract.to_payload()["selected"] == [ref.to_payload() for ref in refs]
    assert [(item.kind, item.id, item.revision) for item in (
        *binding.compiled_agent.selected_mcp,
        *binding.compiled_agent.selected_skills,
        *binding.compiled_agent.selected_tools,
    )] == [
        ("mcp", "server", 6),
        ("skill", "guide", 7),
        ("tool", "a-tool", 5),
        ("tool", "z-tool", 4),
    ]
    restored = compiler.restore(AgentBindingContract.from_payload(binding.binding_contract.to_payload()))
    assert restored.binding_contract == binding.binding_contract
    assert restored.binding_digest == binding.binding_digest


class _AllowAuthorization:
    async def authorize(self, principal: object, action: object, resource: object) -> None:
        del principal, action, resource


class _CaptureSessionExecution:
    def __init__(self) -> None:
        self.request: ExecutionRequest | None = None
        self.agent_id: str | None = None
        self.binding_digest: str | None = None
        self.session_id: str | None = None
        self.session: object | None = None
        self.requires_task_invocation_capture: bool | None = None

    async def start_for_session(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ExecutionRequest,
        *,
        session: object | None = None,
        binding_contract: object | None = None,
        resolved_binding: object | None = None,
        dependency_hold_id: str | None = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> ExecutionHandle:
        self.agent_id = agent_id
        self.binding_digest = binding_digest
        self.session_id = session_id
        self.session = session
        self.request = request
        self.binding_contract = binding_contract
        self.resolved_binding = resolved_binding
        self.dependency_hold_id = dependency_hold_id
        self.requires_task_invocation_capture = requires_task_invocation_capture
        self.budget_scope_id = budget_scope_id
        return ExecutionHandle("execution")


@pytest.mark.asyncio
async def test_session_resume_preserves_mode_planning_and_thinking() -> None:
    service = object.__new__(RuntimeSessionService)
    capture = _CaptureSessionExecution()
    service._authorization = _AllowAuthorization()
    service._execution = capture
    service._admission_execution = capture

    @asynccontextmanager
    async def _consumer(session_id: str, tenant_id: str):
        del session_id, tenant_id
        yield None

    prepared_session = SimpleNamespace(agent_id="agent", cwd=None, status=SessionStatus.OPEN)

    async def _authorized(session_id: str, principal: object, action: object,
                          *, read_session: object = None) -> object:
        del session_id, principal, action, read_session
        return prepared_session

    async def _reconcile(record: object) -> object:
        return record

    service.consume_session = _consumer
    service.read_authorized = _authorized
    service.reconcile_admission = _reconcile

    await service.resume(
        "agent",
        "b" * 64,
        "session",
        ResumeSessionRequest(
            principal=Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value),
            user_prompt="prompt",
            idempotency_key="resume-modes",
            memory_scope=None,
            mode="plan",
            planning=True,
            thinking="high",
        ),
    )

    assert capture.agent_id == "agent"
    assert capture.binding_digest == "b" * 64
    assert capture.session_id == "session"
    assert capture.session is prepared_session
    assert capture.requires_task_invocation_capture is False
    assert capture.request is not None
    assert capture.request.mode == "plan"
    assert capture.request.planning is True
    assert capture.request.thinking == "high"
    assert capture.request.user_prompt == "prompt"
