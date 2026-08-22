#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression coverage for the final Agent composition contract."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from pydantic_ai.capabilities import AbstractCapability

import linktools.ai as ai
from linktools.ai.agent import (
    AgentBindingSnapshot,
    AgentDefinition,
    AgentCatalog,
)
from linktools.ai.agent._output import bind_output
from linktools.ai.capability import RuntimeCapability
from linktools.ai.core import SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    AgentHandle,
    ExecutionHandle,
    ExecutionRequest,
    ResumeSessionRequest,
    Runtime,
    SessionView,
)
from linktools.ai.runtime._factory import _restore_recovery_definitions
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.runtime.state import RecoveryCheckpointState
from linktools.ai.spec import AgentSpec
from linktools.ai.workspace import trusted_workspace_principal


class _DurableCapability(AbstractCapability[None]):
    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return "test-durable-capability"

    @classmethod
    def from_spec(cls, **kwargs: object) -> "_DurableCapability":
        del kwargs
        return cls()


def test_top_level_public_surface_is_exact() -> None:
    assert ai.__all__ == [
        "AgentHandle",
        "AgentSpec",
        "AssetRef",
        "Runtime",
        "RuntimeCapability",
    ]


def test_runtime_capability_from_spec_requires_exact_importable_type() -> None:
    shadow_type = type(
        "_DurableCapability",
        (_DurableCapability,),
        {
            "__module__": __name__,
            "__qualname__": "_DurableCapability",
        },
    )

    with pytest.raises(AIError) as error:
        RuntimeCapability.from_spec("local", shadow_type, config={})

    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_agent_handle_does_not_expose_internal_definition() -> None:
    assert "compile" not in AgentHandle.__dict__


def test_agent_binding_snapshot_is_deeply_immutable() -> None:
    snapshot = AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "model"),
        output_type_module="example.output",
        output_type_qualname="Output",
        output_schema_id="output",
        output_schema_revision=1,
        output_schema_fingerprint="b" * 64,
        local_runtime_capability_descriptors=(
            {"config": {"items": ["first"]}},
        ),
        binding_digest="a" * 64,
    )

    descriptor = snapshot.local_runtime_capability_descriptors[0]
    exposed_config = descriptor["config"]
    assert isinstance(exposed_config, dict)
    exposed_items = exposed_config["items"]
    assert isinstance(exposed_items, list)
    exposed_items.append("mutated")

    persisted = snapshot.to_payload()["local_runtime_capability_descriptors"]
    assert persisted == [{"config": {"items": ["first"]}}]


def test_catalog_uses_durable_semantics_for_restored_capability_instances() -> None:
    digest = "a" * 64
    spec = AgentSpec("agent", 1, "model")
    output = bind_output()
    first_capability = RuntimeCapability.from_spec(
        "local",
        _DurableCapability,
        config={},
    )
    restored_capability = RuntimeCapability.from_spec(
        "local",
        _DurableCapability,
        config={},
    )
    assert first_capability.capability is not restored_capability.capability

    def definition(capability: RuntimeCapability) -> AgentDefinition:
        descriptor = capability.descriptor
        assert descriptor is not None
        snapshot = AgentBindingSnapshot(
            version=1,
            agent_spec=spec,
            output_type_module=output.value_type.__module__,
            output_type_qualname=output.value_type.__qualname__,
            output_schema_id=output.schema_id,
            output_schema_revision=output.schema_revision,
            output_schema_fingerprint=output.schema_fingerprint,
            local_runtime_capability_descriptors=(descriptor,),
            binding_digest=digest,
        )
        return AgentDefinition(
            digest,
            spec,
            SimpleNamespace(fingerprint="b" * 64),
            output,
            (capability,),
            snapshot,
        )

    first = definition(first_capability)
    restored = definition(restored_capability)
    catalog = AgentCatalog({})
    assert catalog.register(first) is first
    assert catalog.register(restored) is first


class _SessionService:
    def __init__(self, binding_digest: str) -> None:
        self._binding_digest = binding_digest

    async def get(self, session_id: str, *, principal: object) -> SessionView:
        del principal
        return SessionView(
            session_id,
            self._binding_digest,
            SessionStatus.OPEN,
        )


class _AllowAuthorization:
    async def authorize(self, principal: object, action: object, resource: object) -> None:
        del principal, action, resource


class _CaptureSessionExecution:
    def __init__(self) -> None:
        self.request: ExecutionRequest | None = None

    async def run_for_session(
        self,
        binding_digest: str,
        session_id: str,
        request: ExecutionRequest,
    ) -> ExecutionHandle:
        del binding_digest, session_id
        self.request = request
        return ExecutionHandle("execution")


@pytest.mark.asyncio
async def test_runtime_session_definition_reports_binding_mismatch() -> None:
    runtime = object.__new__(Runtime)
    runtime.session = _SessionService("b" * 64)
    runtime._catalog = SimpleNamespace(definition=lambda digest: digest)
    preferred = SimpleNamespace(digest="a" * 64)

    with pytest.raises(AIError) as error:
        await runtime._session_definition(
            "session",
            trusted_workspace_principal("tenant"),
            preferred=preferred,
        )

    assert error.value.code is ErrorCode.SESSION_BINDING_MISMATCH


@pytest.mark.asyncio
async def test_runtime_existing_session_reports_binding_mismatch() -> None:
    runtime = object.__new__(Runtime)
    runtime.session = _SessionService("b" * 64)
    definition = SimpleNamespace(
        digest="a" * 64,
        spec=SimpleNamespace(id="agent"),
    )

    with pytest.raises(AIError) as error:
        await runtime._ensure_session(
            definition,
            "session",
            trusted_workspace_principal("tenant"),
        )

    assert error.value.code is ErrorCode.SESSION_BINDING_MISMATCH


@pytest.mark.asyncio
async def test_session_resume_preserves_execution_modes() -> None:
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
        return SimpleNamespace(binding_digest="a" * 64)

    async def _reconcile(record: object) -> object:
        return record

    service._session_consumer = _consumer
    service._authorized = _authorized
    service._reconcile_terminal_admission = _reconcile

    await service.resume(
        "a" * 64,
        "session",
        ResumeSessionRequest(
            principal=trusted_workspace_principal("tenant"),
            user_prompt="prompt",
            idempotency_key="resume-modes",
            planning=True,
            thinking=True,
        ),
    )

    assert capture.request is not None
    assert capture.request.planning is True
    assert capture.request.thinking is True


@pytest.mark.asyncio
async def test_recovery_handoff_schema_must_match_restored_definition() -> None:
    digest = "a" * 64
    snapshot = SimpleNamespace(
        binding_digest=digest,
        agent_spec=SimpleNamespace(id="agent"),
    )
    recovery_input = SimpleNamespace(
        binding_digest=digest,
        planning=False,
        thinking=False,
        binding=snapshot,
        agent_id="agent",
    )
    checkpoint = SimpleNamespace(
        execution_id="execution",
        state=RecoveryCheckpointState.HANDOFF,
        input=recovery_input,
        terminal_handoff=SimpleNamespace(
            outcome=SimpleNamespace(
                output=object(),
                output_schema_id="wrong",
                output_schema_revision=1,
                output_schema_fingerprint="c" * 64,
            )
        ),
    )
    checkpoints = SimpleNamespace(
        list_recoverable_page=lambda **kwargs: None,
    )

    async def _list_recoverable_page(**kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(items=(checkpoint,), next_cursor=None)

    async def _get_execution(*args: object, **kwargs: object) -> None:
        del args, kwargs
        return None

    checkpoints.list_recoverable_page = _list_recoverable_page
    state = SimpleNamespace(
        recovery=SimpleNamespace(checkpoints=checkpoints),
        execution=SimpleNamespace(
            executions=SimpleNamespace(get=_get_execution),
        ),
    )
    definition = SimpleNamespace(
        spec=SimpleNamespace(id="agent"),
        output_binding=SimpleNamespace(
            schema_id="expected",
            schema_revision=1,
            schema_fingerprint="b" * 64,
        ),
    )
    catalog = SimpleNamespace(
        register=lambda value: value,
    )
    compiler = SimpleNamespace(
        restore=lambda value: definition,
    )

    with pytest.raises(AIError) as error:
        await _restore_recovery_definitions(
            catalog,
            compiler,
            state,
            tenant_id="tenant",
        )

    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
