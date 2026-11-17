#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plain Agent composition does not resolve borrowed Task dependencies."""

import pytest
from pydantic_ai.models.test import TestModel
from linktools.ai.core import ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures


class OfflineModel:
    provider = "test"
    model_identity = "test:plain-agent"
    contract = {"version": 3, "provider": provider, "model_identity": model_identity,
                "input_mime_types": [], "settings": {}}

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="answer")


@pytest.mark.asyncio
async def test_plain_agent_does_not_resolve_task_dependencies(monkeypatch) -> None:
    reads = []
    task = Persistence.task.fget
    objects = Persistence.object_store

    def read_task(state):
        reads.append("task")
        return task(state)

    def read_objects(state, domain):
        if domain in {PersistenceDomain.TASK, PersistenceDomain.EVALUATION}:
            reads.append(domain.value)
        return objects(state, domain)

    monkeypatch.setattr(Persistence, "task", property(read_task))
    monkeypatch.setattr(Persistence, "object_store", read_objects)
    features = RuntimeFeatures("plain-agent")
    features.model("default", OfflineModel())
    async with Runtime.open("plain-agent", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        result = await runtime.agents.get("default").run("hello", timeout_seconds=15)
        assert result.result.status is ExecutionStatus.SUCCEEDED
    assert reads == []


@pytest.mark.asyncio
async def test_borrowed_capture_resolves_task_storage_on_demand(monkeypatch) -> None:
    from types import SimpleNamespace
    from linktools.ai.core import Principal, TenantAuthorizationPolicy
    from linktools.ai.inputs import CaptureInputRequest, InputCaptures
    from .test_capture_binding_storage import _seed

    reads = []
    task = Persistence.task.fget
    objects = Persistence.object_store

    def read_task(state):
        reads.append("task")
        return task(state)

    def read_objects(state, domain):
        reads.append(domain)
        return objects(state, domain)

    async with Persistence.open("capture-owner", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        await _seed(state)
        monkeypatch.setattr(Persistence, "task", property(read_task))
        monkeypatch.setattr(Persistence, "object_store", read_objects)
        captures = InputCaptures.from_persistence(state,
            authorization=TenantAuthorizationPolicy("tenant"), execution=SimpleNamespace())
        assert "task" not in reads
        assert PersistenceDomain.TASK not in reads
        assert PersistenceDomain.EVALUATION not in reads
        captured = await captures.capture_input("execution", CaptureInputRequest(
            Principal("principal", "tenant", "service"), "explicit-capture"))
        assert captured is not None
        assert "task" in reads
        assert PersistenceDomain.EVALUATION not in reads


@pytest.mark.asyncio
async def test_task_history_resolves_repository_and_preserves_denial(monkeypatch) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    reads = []
    task = Persistence.task.fget

    def read_task(state):
        reads.append("task")
        return task(state)

    features = RuntimeFeatures("plain-history")
    features.model("default", OfflineModel())
    monkeypatch.setattr(Persistence, "task", property(read_task))
    async with Runtime.open("plain-history", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        assert reads == []
        with pytest.raises(AIError) as error:
            await runtime.history.task_graph("missing", principal=runtime.default_principal)
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert reads == ["task"]


@pytest.mark.asyncio
async def test_task_history_missing_authorization_does_not_resolve_source() -> None:
    from types import SimpleNamespace
    from linktools.ai.core import Principal
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.history import History

    def unexpected_source():
        pytest.fail("Task repository resolved without an authorization policy")

    history = History(SimpleNamespace(), tenant_id="tenant", _task_source=unexpected_source)
    with pytest.raises(AIError) as error:
        await history.task_graph("graph", principal=Principal("reader", "tenant"))
    assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
