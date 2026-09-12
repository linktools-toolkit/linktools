#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression coverage for the final Agent composition contract."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from linktools import ai
from linktools.ai.agent import AgentBindingSnapshot, SemanticPin
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    Agent,
    ExecutionHandle,
    ExecutionRequest,
    ResumeSessionRequest,
)
from linktools.ai.runtime._factory import _restore_recovery_bindings
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.runtime.state._contracts import RecoveryCheckpointState
from linktools.ai.spec import AgentSpec
from linktools.ai.workspace import trusted_workspace_principal


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


def test_agent_binding_snapshot_persists_only_semantic_inputs() -> None:
    snapshot = AgentBindingSnapshot(
        agent_spec=AgentSpec("agent", model="model"),
        base_model={"version": 1, "id": "model"},
        selected=(),
        subagents=(),
        output_mode="structured",
        output_schema={"type": "object", "properties": {"value": {"type": "string"}}},
    )

    payload = snapshot.to_payload()

    assert set(payload) == {
        "agent_spec",
        "base_model",
        "selected",
        "subagents",
        "output_mode",
        "output_schema",
    }
    assert "binding_digest" not in payload
    assert len(snapshot.binding_digest) == 64


def test_semantic_pin_persists_contract_once() -> None:
    pin = SemanticPin(
        "capability",
        "guardrail",
        {"version": 1, "semantic_revision": 3},
    )
    payload = pin.to_payload()

    assert payload == {
        "kind": "capability",
        "id": "guardrail",
        "contract": {"version": 1, "semantic_revision": 3},
    }
    assert SemanticPin.from_payload(payload) == pin
    assert len(pin.fingerprint) == 64

    with pytest.raises(AIError) as error:
        SemanticPin.from_payload({**payload, "fingerprint": pin.fingerprint})
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_agent_binding_snapshot_rejects_unknown_fields() -> None:
    snapshot = AgentBindingSnapshot(
        agent_spec=AgentSpec("agent", model="model"),
        base_model={"version": 1, "id": "model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "object"},
    ).to_payload()
    snapshot["future"] = 3

    with pytest.raises(AIError) as error:
        AgentBindingSnapshot.from_payload(snapshot)

    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


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
    ) -> ExecutionHandle:
        self.agent_id = agent_id
        self.binding_digest = binding_digest
        self.session_id = session_id
        self.request = request
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
        return SimpleNamespace(agent_id="agent")

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
            principal=trusted_workspace_principal("tenant"),
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


@pytest.mark.asyncio
async def test_missing_recovery_execution_fails_closed() -> None:
    checkpoint = SimpleNamespace(
        execution_id="execution",
        state=RecoveryCheckpointState.ADMITTED,
    )

    async def _list_recoverable_page(**kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(items=(checkpoint,), next_cursor=None)

    async def _get_execution(*args: object, **kwargs: object) -> None:
        del args, kwargs
        return None

    state = SimpleNamespace(
        recovery=SimpleNamespace(
            checkpoints=SimpleNamespace(list_recoverable_page=_list_recoverable_page)
        ),
        execution=SimpleNamespace(executions=SimpleNamespace(get=_get_execution)),
    )
    compiler = SimpleNamespace(restore=lambda value: value)
    catalog = SimpleNamespace(
        register_definition=lambda value: value,
        register_binding=lambda value: value,
    )

    with pytest.raises(AIError) as error:
        await _restore_recovery_bindings(
            catalog,
            compiler,
            state,
            tenant_id="tenant",
        )

    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_unavailable_recovery_binding_does_not_block_other_checkpoints() -> None:
    checkpoints = tuple(
        SimpleNamespace(
            execution_id=execution_id,
            state=RecoveryCheckpointState.ADMITTED,
        )
        for execution_id in ("available", "unavailable")
    )
    registered: list[str] = []

    snapshots = {
        execution_id: SimpleNamespace(
            agent_spec=SimpleNamespace(id=execution_id),
            binding_digest=digest,
        )
        for execution_id, digest in (
            ("available", "a" * 64),
            ("unavailable", "b" * 64),
        )
    }
    executions = {
        execution_id: SimpleNamespace(
            execution_id=execution_id,
            binding_digest=snapshot.binding_digest,
            binding=snapshot,
        )
        for execution_id, snapshot in snapshots.items()
    }

    async def _list_recoverable_page(**kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(items=checkpoints, next_cursor=None)

    async def _get_execution(
        execution_id: str,
        *,
        tenant_id: str,
    ) -> object:
        del tenant_id
        return executions[execution_id]

    def _restore(snapshot: object) -> object:
        execution_id = snapshot.agent_spec.id
        if execution_id == "unavailable":
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        return SimpleNamespace(
            digest=snapshot.binding_digest,
            definition=SimpleNamespace(spec=SimpleNamespace(id=execution_id)),
        )

    state = SimpleNamespace(
        recovery=SimpleNamespace(
            checkpoints=SimpleNamespace(list_recoverable_page=_list_recoverable_page)
        ),
        execution=SimpleNamespace(executions=SimpleNamespace(get=_get_execution)),
    )
    catalog = SimpleNamespace(
        register_definition=lambda value: value,
        register_binding=lambda value: registered.append(value.digest) or value,
    )
    compiler = SimpleNamespace(restore=_restore)

    await _restore_recovery_bindings(
        catalog,
        compiler,
        state,
        tenant_id="tenant",
    )

    assert registered == ["a" * 64]
