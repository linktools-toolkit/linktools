#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression coverage for the final Agent composition contract."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from linktools import ai
from linktools.ai.agent import AgentBindingContract, AgentCompiler, CapabilityPin
from linktools.ai.capability import CapabilityContribution, CapabilityGroup, SkillDefinition
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import (
    ExecutionHandle,
    ExecutionRequest,
    ResumeSessionRequest,
)
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.spec import AgentSpec, MCPServerSpec, SkillSpec
from linktools.ai.core import Principal, PrincipalKind
from pydantic_ai.capabilities import AbstractCapability


def test_top_level_public_surface_is_exact() -> None:
    assert ai.__all__ == [
        "Agent",
        "CapabilityGroup",
        "Execution",
        "AgentContext",
        "Runtime",
        "Session",
        "Workspace",
    ]


def test_agent_binding_contract_persists_binding_inputs() -> None:
    binding_contract = AgentBindingContract(
        agent_spec=AgentSpec("agent", model="model"),
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


def test_capability_pin_persists_contract_once() -> None:
    pin = CapabilityPin(
        "capability",
        "guardrail",
        {
            "version": 1,
            "revision": 3,
            "defer_loading": False,
            "config": {},
        },
    )
    payload = pin.to_payload()

    assert payload == {
        "kind": "capability",
        "id": "guardrail",
            "contract": {
                "version": 1,
                "revision": 3,
                "defer_loading": False,
                "config": {},
            },
    }
    assert CapabilityPin.from_payload(payload) == pin
    assert pin.revision == 3

    decoded = CapabilityPin.from_payload({**payload, "future": True})
    assert decoded == pin
    assert "future" not in decoded.to_payload()


def test_agent_binding_contract_preserves_unknown_fields() -> None:
    payload = AgentBindingContract(
        agent_spec=AgentSpec("agent", model="model"),
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
        agent_spec=AgentSpec("agent"),
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
    group = CapabilityGroup("application")
    group.capability(_OrderedCapability("z-capability"), revision=2)
    group.capability(_OrderedCapability("a-capability"), revision=3)

    def business(context: object) -> str:
        del context
        return "ready"

    group.tool(business, name="z-tool", revision=4)
    group.tool(business, name="a-tool", revision=5)
    group.mcp(MCPServerSpec("server", "python", revision=6))
    capture = await group.capture()
    skill = CapabilityContribution.from_declaration(
        SkillDefinition(SkillSpec("guide", "instructions", revision=7))
    )
    spec = AgentSpec("agent")
    compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="test-model").capture(),
        candidates=(*capture.contributions, skill),
        agents={spec.id: spec},
    )
    binding = compiler.bind(compiler.compile(spec))
    pins = binding.binding_contract.selected
    assert [(pin.kind, pin.id, pin.revision) for pin in pins] == [
        ("mcp", "server", 6),
        ("skill", "guide", 7),
        ("tool", "a-tool", 5),
        ("tool", "z-tool", 4),
        ("capability", "z-capability", 2),
        ("capability", "a-capability", 3),
    ]
    expected = {
        (item.kind, item.id): item.contract
        for item in (*capture.contributions, skill)
    }
    assert all(pin.contract == expected[pin.kind, pin.id] for pin in pins)
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
        self.requires_task_invocation_capture: bool | None = None

    async def start_for_session(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ExecutionRequest,
        *,
        binding_contract: object | None = None,
        dependency_hold_id: str | None = None,
        requires_task_invocation_capture: bool = False,
    ) -> ExecutionHandle:
        self.agent_id = agent_id
        self.binding_digest = binding_digest
        self.session_id = session_id
        self.request = request
        self.binding_contract = binding_contract
        self.dependency_hold_id = dependency_hold_id
        self.requires_task_invocation_capture = requires_task_invocation_capture
        return ExecutionHandle("execution")


@pytest.mark.asyncio
async def test_session_resume_preserves_mode_planning_and_thinking() -> None:
    service = object.__new__(DefaultSessionService)
    capture = _CaptureSessionExecution()
    service._authorization = _AllowAuthorization()
    service._execution = capture

    @asynccontextmanager
    async def _consumer(session_id: str, tenant_id: str):
        del session_id, tenant_id
        yield None

    async def _authorized(session_id: str, principal: object, action: object) -> object:
        del session_id, principal, action
        return SimpleNamespace(agent_id="agent", cwd=None)

    async def _reconcile(record: object) -> object:
        return record

    service._session_consumer = _consumer
    service._authorized = _authorized
    service._reconcile_terminal_admission = _reconcile

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
    assert capture.requires_task_invocation_capture is False
    assert capture.request is not None
    assert capture.request.mode == "plan"
    assert capture.request.planning is True
    assert capture.request.thinking == "high"
    assert capture.request.user_prompt == "prompt"
