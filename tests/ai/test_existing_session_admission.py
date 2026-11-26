#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict session admission avoids display reads without permitting lazy creation."""

from unittest.mock import AsyncMock

import pytest

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.session import CreateSessionRequest
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime

from ._runtime_test_helpers import RuntimeUsageModels, model_features

pytestmark = pytest.mark.asyncio


async def test_existing_session_start_uses_session_agent_without_display(monkeypatch):
    features = model_features(RuntimeUsageModels())
    features.agent("other", model="default")
    features.capability(ExecutionCapability(retain_history=True))
    actor = Principal("caller", "default", "service")
    async with Runtime.open("strict-start", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        await runtime.sessions.create("other", CreateSessionRequest(actor, "session", "create"))
        monkeypatch.setattr(runtime._session_service, "get", AsyncMock(side_effect=AssertionError("no display read")))
        run = await runtime.sessions.start("session", "hello", principal=actor, idempotency_key="first")
        await run.wait(timeout_seconds=10)
        assert (await run.inspect()).agent_id == "other"
        replay = await runtime.sessions.start("session", "hello", principal=actor, idempotency_key="first")
        assert replay.execution_id == run.execution_id


async def test_existing_session_start_does_not_create_missing_or_unauthorized(monkeypatch):
    actor = Principal("caller", "default", "service")
    other = Principal("other", "default", "service")
    async with Runtime.open("strict-missing", storage=PersistenceStorage.in_memory(),
                            features=model_features(RuntimeUsageModels())) as runtime:
        await runtime.sessions.create("default", CreateSessionRequest(other, "private", "create"))
        create = AsyncMock(side_effect=AssertionError("strict admission cannot create"))
        monkeypatch.setattr(runtime._session_service, "create", create)
        for identity in ("missing", "private"):
            with pytest.raises(AIError) as caught:
                await runtime.sessions.start(identity, "hello", principal=actor, idempotency_key=identity)
            assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        create.assert_not_awaited()


async def test_existing_session_close_during_binding_still_fails_cas(monkeypatch):
    actor = Principal("caller", "default", "service")
    async with Runtime.open("strict-race", storage=PersistenceStorage.in_memory(),
                            features=model_features(RuntimeUsageModels())) as runtime:
        session = await runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        resolve = runtime._binding_resolver.resolve_agent

        async def close_before_resolve(*args, **kwargs):
            await session.close(idempotency_key="close")
            return await resolve(*args, **kwargs)

        monkeypatch.setattr(runtime._binding_resolver, "resolve_agent", close_before_resolve)
        with pytest.raises(AIError) as caught:
            await runtime.sessions.start("session", "hello", principal=actor, idempotency_key="turn")
        assert caught.value.code in {ErrorCode.SESSION_CONFLICT, ErrorCode.STORAGE_CONFLICT}
        assert str((await session.inspect()).status) == "CLOSED"


async def test_existing_session_start_after_same_version_reopen(tmp_path):
    class HostAuthorization(TenantAuthorizationPolicy):
        """Supply the host policy on both sides of a physical reopen."""

    actor = Principal("caller", "default", "service")
    storage = PersistenceStorage.from_root(tmp_path / "state")
    features = model_features(RuntimeUsageModels())
    async with Runtime.open("strict-reopen", storage=storage, features=features,
                            authorization=HostAuthorization()) as runtime:
        await runtime.sessions.create(CreateSessionRequest(actor, "session", "create"))
    async with Runtime.open("strict-reopen", storage=storage, features=features,
                            authorization=HostAuthorization()) as runtime:
        turn = await runtime.sessions.start(
            "session", "hello", principal=actor, agent=runtime.agents.get(), idempotency_key="turn",
        )
        result = await turn.wait_final(timeout_seconds=10)
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.key.session_id == "session"
        assert await runtime._state.conversation.sessions.get("session", tenant_id=actor.tenant_id) is None


async def test_existing_session_explicit_output_and_correlation():
    from pydantic import BaseModel
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from ._runtime_test_helpers import _Models, _UsageFunctionModel

    class Answer(BaseModel):
        count: int

    async def model(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"count": 7})])

    actor = Principal("caller", "default", "service")
    async with Runtime.open("strict-output", storage=PersistenceStorage.in_memory(),
                            features=model_features(_Models(_UsageFunctionModel(model)))) as runtime:
        await runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        execution = await runtime.sessions.start("session", "count", principal=actor,
            idempotency_key="turn", output=Answer, correlation={"feature": "qa"})
        await execution.wait(timeout_seconds=10)
        assert (await execution.result()).output == {"count": 7}
        stored = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=actor.tenant_id)
        assert stored.correlation["feature"] == "qa"


async def test_existing_session_uses_binding_policy_and_reads_live_source_next_time(monkeypatch):
    from linktools.ai.capability import SessionCapability

    actor = Principal("caller", "default", "service")
    features = model_features(RuntimeUsageModels())
    disabled_features = model_features(RuntimeUsageModels())
    disabled_features.capability(SessionCapability(enabled=False), scope="runtime")
    async with Runtime.open("strict-source", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        await runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        resolve = features.resolve
        calls = []
        enabled = False

        async def current():
            calls.append(enabled)
            return await resolve() if enabled else await disabled_features.resolve()

        monkeypatch.setattr(features, "resolve", current)
        with pytest.raises(AIError) as caught:
            await runtime.sessions.start("session", "hello", principal=actor, idempotency_key="turn")
        assert caught.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
        assert calls == [False]
        enabled = True
        calls.clear()
        execution = await runtime.sessions.start("session", "hello", principal=actor, idempotency_key="turn")
        assert calls == [True]
        await execution.wait(timeout_seconds=10)


async def test_existing_session_pins_rules_from_same_declaration(monkeypatch):
    from dataclasses import replace
    from linktools.ai.spec import RepositoryInstructionDocument, RepositoryInstructions

    actor = Principal("caller", "default", "service")
    features = model_features(RuntimeUsageModels())
    async with Runtime.open("strict-rules", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        await runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        resolve = features.resolve
        captures = []

        async def current():
            resolution = await resolve()
            text = f"declaration-{len(captures) + 1}"
            captures.append(text)
            return replace(resolution, instructions=RepositoryInstructions((
                RepositoryInstructionDocument("rule:current", ".", text),)))

        monkeypatch.setattr(features, "resolve", current)
        execution = await runtime.sessions.start("session", "hello", principal=actor, idempotency_key="turn")
        assert captures == ["declaration-1"]
        await execution.wait(timeout_seconds=10)
        backend = runtime._execution_service.runtime_backend()
        record = await backend.load_execution(execution.execution_id, tenant_id=runtime.tenant_id)
        accepted = await backend.load_input_context(record.input_context)
        documents = RepositoryInstructions.from_payload(accepted.repository_instructions).documents
        assert tuple(document.content for document in documents) == ("declaration-1",)


async def test_existing_session_still_reads_workspace_at_admission(tmp_path, monkeypatch):
    from linktools.ai import Workspace
    from linktools.ai.spec import RepositoryInstructions
    from ._runtime_test_helpers import workspace_features

    rules = tmp_path / "AGENTS.md"
    rules.write_text("before binding", encoding="utf-8")
    actor = Principal("caller", "default", "service")
    features = model_features(RuntimeUsageModels(), workspace_features("strict-files", workspace=Workspace.load(tmp_path)))
    async with Runtime.open("strict-files", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        await runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        resolve = runtime._binding_resolver.resolve_agent

        async def change_workspace(*args, **kwargs):
            binding = await resolve(*args, **kwargs)
            rules.write_text("after binding", encoding="utf-8")
            return binding

        monkeypatch.setattr(runtime._binding_resolver, "resolve_agent", change_workspace)
        execution = await runtime.sessions.start("session", "hello", principal=actor, idempotency_key="turn")
        await execution.wait(timeout_seconds=10)
        backend = runtime._execution_service.runtime_backend()
        record = await backend.load_execution(execution.execution_id, tenant_id=runtime.tenant_id)
        accepted = await backend.load_input_context(record.input_context)
        documents = RepositoryInstructions.from_payload(accepted.repository_instructions).documents
        assert tuple(document.content for document in documents) == ("after binding",)


async def test_existing_session_custom_adapter_uses_authoritative_record_reader(monkeypatch):
    from linktools.ai.runtime._session import _SessionExecutionAdapter

    actor = Principal("caller", "default", "service")
    async with Runtime.open("strict-custom", storage=PersistenceStorage.in_memory(),
                            features=model_features(RuntimeUsageModels())) as runtime:
        await runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        service = runtime._session_service
        owner = service._admission_execution
        calls = []

        class CustomExecution:
            async def start_for_session(self, *args, **kwargs):
                calls.append((args, kwargs))
                return await owner.start_for_session(*args, **kwargs)

        custom = CustomExecution()
        monkeypatch.setattr(service, "_admission_execution", custom)
        monkeypatch.setattr(service, "_execution", _SessionExecutionAdapter(custom))
        read = AsyncMock(wraps=service._read_session)
        monkeypatch.setattr(service, "_read_session", read)
        execution = await runtime.sessions.start("session", "hello", principal=actor, idempotency_key="turn")
        read.assert_awaited_once_with("session", tenant_id=actor.tenant_id)
        assert len(calls) == 1
        assert "_start_read" not in calls[0][1]
        await execution.wait(timeout_seconds=10)
        with pytest.raises(AIError) as caught:
            await runtime.sessions.start("missing", "hello", principal=actor, idempotency_key="missing")
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert len(calls) == 1


async def test_existing_session_concurrent_same_key_converges(monkeypatch):
    import asyncio

    actor = Principal("caller", "default", "service")
    async with Runtime.open("strict-concurrent", storage=PersistenceStorage.in_memory(),
                            features=model_features(RuntimeUsageModels())) as runtime:
        await runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        resolve = runtime._binding_resolver.resolve_agent
        ready = asyncio.Event()
        arrivals = 0

        async def together(*args, **kwargs):
            nonlocal arrivals
            binding = await resolve(*args, **kwargs)
            arrivals += 1
            if arrivals == 2:
                ready.set()
            await ready.wait()
            return binding

        monkeypatch.setattr(runtime._binding_resolver, "resolve_agent", together)
        first, second = await asyncio.wait_for(asyncio.gather(*(
            runtime.sessions.start("session", "hello", principal=actor, idempotency_key="same")
            for _ in range(2)
        )), 10)
        assert first.execution_id == second.execution_id
        await first.wait(timeout_seconds=10)


async def test_existing_session_rechecks_execution_permission_after_binding(monkeypatch):
    from linktools.ai.core import AuthorizationAction, TenantAuthorizationPolicy

    class Policy(TenantAuthorizationPolicy):
        revoked = False

        async def authorize(self, principal, action, resource):
            if self.revoked and action is AuthorizationAction.EXECUTION_RUN:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await super().authorize(principal, action, resource)

    actor = Principal("caller", "default", "service")
    policy = Policy()
    async with Runtime.open("strict-revocation", storage=PersistenceStorage.in_memory(),
                            features=model_features(RuntimeUsageModels()), authorization=policy) as runtime:
        await runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        resolve = runtime._binding_resolver.resolve_agent

        async def revoke(*args, **kwargs):
            binding = await resolve(*args, **kwargs)
            policy.revoked = True
            return binding

        monkeypatch.setattr(runtime._binding_resolver, "resolve_agent", revoke)
        with pytest.raises(AIError) as caught:
            await runtime.sessions.start("session", "hello", principal=actor, idempotency_key="turn")
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert (await runtime.sessions.inspect("session", principal=actor)).active_execution_id is None
