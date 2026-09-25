#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression coverage for the final Agent composition contract."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from linktools import ai
from linktools.ai.agent import AgentBindingContract, CapabilityPin
from linktools.ai.runtime import (
    Agent,
    ExecutionHandle,
    ExecutionRequest,
    ResumeSessionRequest,
)
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.spec import AgentSpec
from linktools.ai.core import Principal, PrincipalKind


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


def test_runtime_bound_agent_does_not_expose_compile_or_registration() -> None:
    assert "compile" not in Agent.__dict__
    assert "register" not in Agent.__dict__
    assert "define" not in Agent.__dict__


def test_agent_binding_contract_persists_binding_inputs() -> None:
    binding_contract = AgentBindingContract(
        agent_spec=AgentSpec("agent", model_route="model"),
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
        "runtime_capability",
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
        "kind": "runtime_capability",
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
        agent_spec=AgentSpec("agent", model_route="model"),
        model_contract={"version": 1, "id": "model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "object"},
    ).to_payload()
    payload["future"] = 3

    decoded = AgentBindingContract.from_payload(payload)

    assert decoded.to_payload()["future"] == 3


class _AllowAuthorization:
    async def authorize(self, principal: object, action: object, resource: object) -> None:
        del principal, action, resource


class _CaptureSessionExecution:
    def __init__(self) -> None:
        self.request: ExecutionRequest | None = None
        self.agent_id: str | None = None
        self.binding_digest: str | None = None
        self.session_id: str | None = None

    async def start_for_session(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ExecutionRequest,
        *,
        binding_contract: object | None = None,
    ) -> ExecutionHandle:
        self.agent_id = agent_id
        self.binding_digest = binding_digest
        self.session_id = session_id
        self.request = request
        self.binding_contract = binding_contract
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
    assert capture.request is not None
    assert capture.request.mode == "plan"
    assert capture.request.planning is True
    assert capture.request.thinking == "high"
    assert capture.request.user_prompt == "prompt"
