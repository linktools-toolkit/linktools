#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression coverage for the final Agent composition contract."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.capability import RuntimeCapability
from linktools.ai.core import (
    HmacCursorSigner,
    SessionStatus,
    TenantAuthorizationPolicy,
    step_conversation_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    AgentHandle,
    ExecutionHandle,
    ExecutionRequest,
    ResumeSessionRequest,
    Runtime,
    RuntimeDomain,
    RuntimeState,
    SessionView,
)
from linktools.ai.runtime._factory import _restore_recovery_bindings
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.runtime.state import RecoveryCheckpointState
from linktools.ai.runtime.state._contracts import ConversationCursor, SessionRecord
from linktools.ai.spec import AgentSpec
from linktools.ai.workspace import trusted_workspace_principal
from pydantic import BaseModel
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord

from linktools import ai


class CompositionStructuredOutput(BaseModel):
    value: str


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


def test_runtime_capability_from_spec_uses_serialization_identity() -> None:
    class RelocatableCapability(_DurableCapability):
        @classmethod
        def get_serialization_name(cls) -> "str | None":
            return "test-relocatable-capability"

    relocated_type = type(
        "RelocatedCapability",
        (RelocatableCapability,),
        {
            "__module__": "example.moved",
            "__qualname__": "RelocatedCapability",
        },
    )
    value = RuntimeCapability.from_spec("local", relocated_type, config={})
    descriptor = value.descriptor
    assert descriptor is not None
    assert descriptor["serialization_name"] == "test-relocatable-capability"
    assert set(descriptor) == {
        "id",
        "revision",
        "serialization_name",
        "config",
        "fingerprint",
    }



def test_agent_handle_does_not_expose_internal_definition() -> None:
    assert "compile" not in AgentHandle.__dict__


def test_agent_binding_snapshot_is_deeply_immutable() -> None:
    snapshot = AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "model"),
        agent_digest="c" * 64,
        output_schema_id="output",
        output_schema_revision=1,
        output_schema_fingerprint="b" * 64,
        local_runtime_capability_descriptors=(
            {"config": {"items": ["first"]}},
        ),
        binding_digest="a" * 64,
        global_runtime_capability_descriptors=(),
    )

    descriptor = snapshot.local_runtime_capability_descriptors[0]
    exposed_config = descriptor["config"]
    assert isinstance(exposed_config, dict)
    exposed_items = exposed_config["items"]
    assert isinstance(exposed_items, list)
    exposed_items.append("mutated")

    persisted = snapshot.to_payload()["local_runtime_capability_descriptors"]
    assert persisted == [{"config": {"items": ["first"]}}]


def test_agent_definition_no_longer_owns_output_binding() -> None:
    from dataclasses import fields

    from linktools.ai.agent import AgentDefinition

    names = {field.name for field in fields(AgentDefinition)}
    assert "digest" in names
    assert "spec" in names
    assert "output_binding" not in names
    assert "binding_snapshot" not in names

class _SessionService:
    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id

    async def get(self, session_id: str, *, principal: object) -> SessionView:
        del principal
        return SessionView(
            session_id,
            self._agent_id,
            SessionStatus.OPEN,
        )


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
async def test_runtime_existing_session_reports_agent_mismatch() -> None:
    runtime = object.__new__(Runtime)
    runtime.session = _SessionService("other-agent")
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


_MICRO_CHANGES = (
    "revision",
    "system_prompt",
    "instructions",
    "allow_tools",
    "usage_limits",
    "model",
    "skill",
    "capability",
    "output",
    "catalog",
    "platform_policy",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    _MICRO_CHANGES,
)
async def test_durable_session_survives_agent_micro_changes(tmp_path, change: str) -> None:
    root = tmp_path / "runtime"
    namespace = f"micro-change-{change}"
    state = RuntimeState.filesystem(root)
    await state.initialize(namespace=namespace, tenant_id="tenant")
    now = datetime.now(timezone.utc)
    try:
        created = await state.conversation.sessions.create(
            SessionRecord(
                session_id="session",
                tenant_id="tenant",
                owner_principal_id="workspace",
                status=SessionStatus.OPEN,
                revision=0,
                resource_generation=0,
                cwd=None,
                metadata={},
                created_at=now,
                updated_at=now,
                closed_at=None,
                active_execution_id=None,
                continuation=None,
                history_id=None,
                agent_id="agent",
            )
        )
        await state.conversation.sessions.admit_execution(
            "session",
            tenant_id="tenant",
            execution_id="turn-execution",
            expected=None,
        )
        conversation_id = step_conversation_id(
            namespace=namespace,
            tenant_id="tenant",
            execution_id="turn-run",
        )
        turn_messages = (
            ModelRequest(
                parts=[UserPromptPart(content="first prompt")],
                conversation_id=conversation_id,
            ),
            ModelResponse(
                parts=[TextPart(content="first answer")],
                conversation_id=conversation_id,
            ),
        )
        await state.steps.register_run(
            RunRecord(
                run_id="turn-run",
                conversation_id=conversation_id,
                parent_run_id=None,
                agent_name="agent",
                metadata={},
                started_at=now,
            )
        )
        await state.steps.save_snapshot(
            ContinuableSnapshot(
                run_id="turn-run",
                step_index=len(turn_messages),
                messages=list(turn_messages),
                conversation_id=conversation_id,
                parent_run_id=None,
                agent_name="agent",
                timestamp=now,
                state="complete",
            )
        )
        await state.steps.materialize_conversation(step_run_id="turn-run")
        await state.conversation.sessions.advance_continuation(
            "session",
            tenant_id="tenant",
            execution_id="turn-execution",
            expected=None,
            next_cursor=ConversationCursor("turn-run", history_id=created.history_id),
        )
        await state.conversation.sessions.release_execution(
            "session",
            tenant_id="tenant",
            execution_id="turn-execution",
        )
    finally:
        await state.close()

    reopened = RuntimeState.filesystem(root)
    await reopened.initialize(namespace=namespace, tenant_id="tenant")
    try:
        service = DefaultSessionService(
            reopened.conversation,
            reopened.execution.executions,
            TenantAuthorizationPolicy(),
            object(),
            HmacCursorSigner("session", b"session-key"),
            history_reader=object(),
        )
        capture = _CaptureSessionExecution()
        service._gated_execution = capture
        runtime = object.__new__(Runtime)
        runtime._catalog = SimpleNamespace(
            definition=lambda digest: SimpleNamespace(
                digest=digest,
                spec=SimpleNamespace(id="agent", changed_input=change),
            ),
            register_binding=lambda binding: binding,
        )
        runtime._compiler = SimpleNamespace(
            bind=lambda definition, output: SimpleNamespace(
                digest=binding_digest,
                definition=definition,
                output=output,
            )
        )
        runtime._closed = False
        runtime._local_coordinator = None
        runtime.session = service
        binding_digest = f"{1 + _MICRO_CHANGES.index(change):064x}"
        old_binding_digest = "0" * 64
        assert binding_digest != old_binding_digest
        await runtime._start_for_agent(
            "agent-definition",
            "next prompt",
            output=None,
            principal=trusted_workspace_principal("tenant"),
            session_id="session",
            idempotency_key=f"micro-change-{change}",
            memory_scope=None,
            planning=False,
            thinking=False,
        )
        view = await service.get(
            "session",
            principal=trusted_workspace_principal("tenant"),
        )
        record = await reopened.conversation.sessions.get("session", tenant_id="tenant")
        conversation_store = reopened.steps.read_store(RuntimeDomain.CONVERSATION)
        assert await conversation_store.load_model_context(run_id="turn-run") == turn_messages
        assert capture.agent_id == "agent"
        assert capture.binding_digest == binding_digest
        assert capture.session_id == "session"
        assert view.agent_id == "agent"
        assert record is not None
        assert record.agent_id == "agent"
        assert record.continuation == ConversationCursor(
            "turn-run",
            history_id=record.history_id,
        )
    finally:
        await reopened.close()


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
            idempotency_key="resume-modes",
            planning=True,
            thinking=True,
        ),
    )

    assert capture.request is not None
    assert capture.request.planning is True
    assert capture.request.thinking is True


@pytest.mark.asyncio
async def test_recovery_handoff_schema_must_match_restored_binding() -> None:
    binding_digest = "a" * 64
    snapshot = SimpleNamespace(binding_digest=binding_digest)
    recovery_input = SimpleNamespace(
        binding_digest=binding_digest,
        planning=False,
        thinking=False,
        binding=snapshot,
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

    async def _list_recoverable_page(**kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(items=(checkpoint,), next_cursor=None)

    async def _get_execution(*args: object, **kwargs: object) -> None:
        del args, kwargs

    state = SimpleNamespace(
        recovery=SimpleNamespace(
            checkpoints=SimpleNamespace(list_recoverable_page=_list_recoverable_page)
        ),
        execution=SimpleNamespace(
            executions=SimpleNamespace(get=_get_execution),
        ),
    )
    definition = SimpleNamespace(digest="d" * 64)
    binding = SimpleNamespace(
        digest=binding_digest,
        definition=definition,
        output_binding=SimpleNamespace(
            schema_id="expected",
            schema_revision=1,
            schema_fingerprint="b" * 64,
        ),
    )
    catalog = SimpleNamespace(
        register_definition=lambda value: value,
        register_binding=lambda value: value,
    )
    compiler = SimpleNamespace(restore=lambda value: binding)

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
    available_digest = "a" * 64
    unavailable_digest = "b" * 64
    checkpoints = tuple(
        SimpleNamespace(
            execution_id=execution_id,
            state=RecoveryCheckpointState.ADMITTED,
            terminal_handoff=None,
            input=SimpleNamespace(
                binding_digest=digest,
                planning=False,
                thinking=False,
                binding=SimpleNamespace(
                    agent_spec=SimpleNamespace(id=execution_id),
                ),
            ),
        )
        for execution_id, digest in (
            ("available", available_digest),
            ("unavailable", unavailable_digest),
        )
    )
    restored: list[str] = []
    registered: list[str] = []

    async def _list_recoverable_page(**kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(items=checkpoints, next_cursor=None)

    async def _get_execution(*args: object, **kwargs: object) -> None:
        del args, kwargs

    def _restore(snapshot: object) -> object:
        execution_id = snapshot.agent_spec.id
        restored.append(execution_id)
        if execution_id == "unavailable":
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        return SimpleNamespace(
            digest=available_digest,
            definition=SimpleNamespace(spec=SimpleNamespace(id=execution_id)),
        )

    state = SimpleNamespace(
        recovery=SimpleNamespace(
            checkpoints=SimpleNamespace(list_recoverable_page=_list_recoverable_page)
        ),
        execution=SimpleNamespace(
            executions=SimpleNamespace(get=_get_execution),
        ),
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

    assert restored == ["available", "unavailable"]
    assert registered == [available_digest]


@pytest.mark.asyncio
async def test_startup_recovery_isolates_one_unavailable_execution(caplog: pytest.LogCaptureFixture) -> None:
    checkpoints = tuple(
        SimpleNamespace(
            execution_id=execution_id,
            state=RecoveryCheckpointState.ADMITTED,
        )
        for execution_id in ("available", "unavailable")
    )
    processed: list[str] = []
    backend = object.__new__(LocalExecutionBackend)
    backend._recovery_enabled = True
    backend._tenant_id = "tenant"

    async def list_page(**kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(items=checkpoints, next_cursor=None)

    backend._recovery = SimpleNamespace(
        checkpoints=SimpleNamespace(
            list_recoverable_page=list_page,
        )
    )

    async def reconcile(checkpoint: object) -> None:
        execution_id = checkpoint.execution_id
        processed.append(execution_id)
        if execution_id == "unavailable":
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)

    backend._reconcile_checkpoint = reconcile
    caplog.set_level(logging.WARNING)

    await backend.reconcile()

    assert processed == ["available", "unavailable"]
    assert any("unavailable" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_startup_recovery_does_not_suppress_storage_integrity() -> None:
    checkpoint = SimpleNamespace(
        execution_id="corrupt",
        state=RecoveryCheckpointState.ADMITTED,
    )
    backend = object.__new__(LocalExecutionBackend)
    backend._recovery_enabled = True
    backend._tenant_id = "tenant"

    async def list_page(**kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(items=(checkpoint,), next_cursor=None)

    backend._recovery = SimpleNamespace(
        checkpoints=SimpleNamespace(
            list_recoverable_page=list_page,
        )
    )

    async def reconcile(_checkpoint: object) -> None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    backend._reconcile_checkpoint = reconcile

    with pytest.raises(AIError) as error:
        await backend.reconcile()

    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_agent_and_binding_identity_split_acceptance(tmp_path) -> None:
    from linktools.ai.model import ModelRegistry
    from linktools.ai.workspace import Workspace, open_workspace_runtime

    workspace = Workspace.load(tmp_path)
    models = ModelRegistry.openai(model="gpt-test")
    async with open_workspace_runtime(
        workspace,
        models=models,
        outputs=(bind_output(CompositionStructuredOutput),),
        runtime_capability_types=(_DurableCapability,),
    ) as runtime:
        base = runtime.agent()
        text_binding = runtime._bind_agent(base._agent_digest)
        structured_binding = runtime._bind_agent(
            base._agent_digest,
            output=CompositionStructuredOutput,
        )

        assert text_binding.definition.digest == structured_binding.definition.digest
        assert text_binding.digest != structured_binding.digest
        assert runtime._bind_agent(base._agent_digest).digest == text_binding.digest

        local_capability = RuntimeCapability.from_spec(
            "local-identity",
            _DurableCapability,
            config={"mode": "strict"},
            revision=1,
        )
        local = runtime.agent(capabilities=(local_capability,))
        local_binding = runtime._bind_agent(local._agent_digest)
        assert local._agent_digest != base._agent_digest
        assert local_binding.digest != text_binding.digest

        session = await base.create_session("identity-session")
        assert session.agent_id == base.agent_id
        await runtime._ensure_session(
            runtime._definition(base._agent_digest),
            session.session_id,
            runtime.default_principal,
        )
        await runtime._ensure_session(
            runtime._definition(local._agent_digest),
            session.session_id,
            runtime.default_principal,
        )
