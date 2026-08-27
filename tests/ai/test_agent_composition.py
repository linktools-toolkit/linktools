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
from linktools.ai.runtime.state import RecoveryCheckpointState
from linktools.ai.spec import AgentSpec
from linktools.ai.workspace import trusted_workspace_principal


def test_top_level_public_surface_is_exact() -> None:
    assert ai.__all__ == [
        "Agent",
        "CapabilityGroup",
        "Execution",
        "RunContext",
        "Runtime",
        "Session",
        "Workspace",
    ]


def test_runtime_bound_agent_does_not_expose_compile_or_registration() -> None:
    assert "compile" not in Agent.__dict__
    assert "register" not in Agent.__dict__
    assert "define" not in Agent.__dict__


def test_agent_binding_snapshot_persists_only_final_v1_identity_contract() -> None:
    snapshot = AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", model="model"),
        model={"version": 1, "id": "model"},
        selected=(),
        subagents=(),
        output_mode="structured",
        output_schema={"type": "object", "properties": {"value": {"type": "string"}}},
        binding_digest="a" * 64,
    )

    payload = snapshot.to_payload()

    assert set(payload) == {
        "version",
        "agent_spec",
        "model",
        "selected",
        "subagents",
        "output_mode",
        "output_schema",
        "binding_digest",
    }
    assert "agent_digest" not in payload
    assert "output_schema_id" not in payload
    assert "output_schema_revision" not in payload
    assert "output_schema_fingerprint" not in payload
    assert "binding_fingerprint" not in payload
    assert "runtime_capabilities" not in payload


def test_semantic_pin_persists_historical_source_without_fingerprint() -> None:
    pin = SemanticPin(
        "capability",
        "guardrail",
        1,
        {"version": 1, "semantic_revision": 3},
    )
    payload = pin.to_payload()

    assert payload == {
        "kind": "capability",
        "id": "guardrail",
        "contract_version": 1,
        "contract": {"version": 1, "semantic_revision": 3},
    }
    assert SemanticPin.from_payload(payload) == pin
    assert len(pin.fingerprint) == 64

    legacy = dict(payload)
    legacy["fingerprint"] = pin.fingerprint
    with pytest.raises(AIError) as error:
        SemanticPin.from_payload(legacy)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_agent_binding_snapshot_rejects_unknown_version() -> None:
    snapshot = AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", model="model"),
        model={"version": 1, "id": "model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "object"},
        binding_digest="a" * 64,
    ).to_payload()
    snapshot["version"] = 2

    with pytest.raises(AIError) as error:
        AgentBindingSnapshot.from_payload(snapshot)

    assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


class _AllowAuthorization:
    async def authorize(self, principal: object, action: object, resource: object) -> None:
        del principal, action, resource


class _CaptureSessionExecution:
    def __init__(self) -> None:
        self.request: ExecutionRequest | None = None
        self.agent_id: str | None = None
        self.binding_digest: str | None = None
        self.session_id: str | None = None

    async def run_for_session(
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
    service._gated_execution = capture

    @asynccontextmanager
    async def _consumer(session_id: str, tenant_id: str):
        del session_id, tenant_id
        yield None

    async def _authorized(session_id: str, principal: object, action: object) -> object:
        del session_id, principal, action
        return SimpleNamespace(agent_id="agent", metadata={})

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
            user_prompt_codec="text",
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
    assert capture.request.user_prompt_codec == "text"


@pytest.mark.asyncio
async def test_recovery_binding_digest_mismatch_fails_closed() -> None:
    snapshot = SimpleNamespace(binding_digest="a" * 64)
    checkpoint = SimpleNamespace(
        execution_id="execution",
        state=RecoveryCheckpointState.ADMITTED,
        input=SimpleNamespace(
            binding_digest="a" * 64,
            mode="run",
            planning=False,
            thinking=False,
            binding=snapshot,
        ),
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
    compiler = SimpleNamespace(
        restore=lambda value: SimpleNamespace(
            digest="b" * 64,
            definition=SimpleNamespace(spec=SimpleNamespace(id="agent")),
        )
    )
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
            input=SimpleNamespace(
                binding_digest=digest,
                mode="run",
                planning=False,
                thinking=False,
                binding=SimpleNamespace(agent_spec=SimpleNamespace(id=execution_id)),
            ),
        )
        for execution_id, digest in (
            ("available", "a" * 64),
            ("unavailable", "b" * 64),
        )
    )
    registered: list[str] = []

    async def _list_recoverable_page(**kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(items=checkpoints, next_cursor=None)

    async def _get_execution(*args: object, **kwargs: object) -> None:
        del args, kwargs
        return None

    def _restore(snapshot: object) -> object:
        execution_id = snapshot.agent_spec.id
        if execution_id == "unavailable":
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        return SimpleNamespace(
            digest="a" * 64,
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
