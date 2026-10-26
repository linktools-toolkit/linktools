#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.capability import AgentContext
from linktools.ai.model import create_model
from linktools.ai.agent._model_resolver import _ModelResolver
from linktools.ai.runtime import RuntimeFeatures


@pytest.mark.asyncio
async def test_model_capture_is_lazy_isolated_and_success_is_shared() -> None:
    calls = []
    binding = create_model(protocol="openai", model="fixture")
    def load():
        calls.append("load")
        return binding
    features = RuntimeFeatures()
    features.model("assistant", load)
    captured = await features.capture()
    features.model("assistant", create_model(protocol="anthropic", model="replacement"))
    assert calls == []
    resolved = _ModelResolver(captured.models).resolve("assistant")
    assert calls == []
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(lambda _: resolved.model_identity, range(8)))
    assert identities == ["openai:fixture"] * 8
    assert calls == ["load"]
    assert dict(resolved.contract) == dict(binding.contract)
    assert _ModelResolver((await features.capture()).models).resolve("assistant").model_identity == "anthropic:replacement"
    assert calls == ["load"]


@pytest.mark.asyncio
async def test_unused_model_loader_is_not_called() -> None:
    features = RuntimeFeatures()
    def fail():
        pytest.fail("unused loader called")
    features.model("unused", fail)
    await features.capture()


def test_lazy_model_failure_can_be_retried_and_async_result_is_rejected() -> None:
    attempts = []
    def load():
        attempts.append(1)
        if len(attempts) == 1:
            raise AIError(ErrorCode.MODEL_CONFIG_INVALID, "fixture configuration missing")
        return create_model(protocol="openai", model="fixture")
    binding = _ModelResolver({"model": load}).resolve("model")
    with pytest.raises(AIError):
        _ = binding.contract
    assert binding.model_identity == "openai:fixture"
    assert len(attempts) == 2
    async def async_load():
        return create_model(protocol="openai", model="fixture")
    with pytest.raises(TypeError):
        _ModelResolver({"async": async_load})
    hidden = _ModelResolver({"async": lambda: async_load()}).resolve("async")
    with pytest.raises(TypeError):
        _ = hidden.contract


def test_lazy_model_restore_uses_registered_route_and_rejects_missing_route() -> None:
    binding = create_model(protocol="openai", model="fixture")
    resolver = _ModelResolver({"assistant": lambda: binding})
    assert resolver.resolve("assistant").model_identity == binding.model_identity
    assert resolver.resolve("assistant").model_identity == binding.model_identity
    with pytest.raises(AIError) as error:
        resolver.resolve("missing")
    assert error.value.code == ErrorCode.MODEL_CONNECTION_NOT_FOUND


def test_recursive_loader_fails_without_deadlocking() -> None:
    resolver = _ModelResolver({"loop": lambda: resolver.resolve("loop").contract})
    with pytest.raises(AIError):
        _ = resolver.resolve("loop").contract

@pytest.mark.asyncio
async def test_runtime_open_compiles_routes_without_eager_model_loading() -> None:
    from linktools.ai.runtime import Runtime, RuntimeStorage
    calls = []
    def load():
        calls.append(1)
        return create_model(protocol="openai", model="fixture")
    features = RuntimeFeatures()
    features.model("default", load)
    features.model("unused", lambda: pytest.fail("unused model loaded"))
    features.agent("assistant", model="default")
    async with Runtime.open("lazy-model-test", storage=RuntimeStorage.in_memory(), features=features) as runtime:
        assert calls == []
        binding = runtime.agents.get("assistant").model
        assert calls == [1]
        assert binding.model_identity == "openai:fixture"
        assert calls == [1]


@pytest.mark.asyncio
async def test_callable_custom_binding_is_not_mistaken_for_a_loader() -> None:
    reads = []
    class CallableBinding:
        @property
        def provider(self):
            reads.append("provider")
            return "custom"
        @property
        def model_identity(self):
            reads.append("identity")
            return "custom:fixture"
        @property
        def contract(self):
            reads.append("contract")
            return {"provider": "custom", "model_identity": "custom:fixture"}
        def materialize(self):
            pytest.fail("client created during capture")
        def __call__(self):
            pytest.fail("binding called as loader")
    binding = CallableBinding()
    features = RuntimeFeatures()
    features.model("custom", binding)
    capture = await features.capture()
    assert reads == []
    assert _ModelResolver(capture.models).resolve("custom") is binding
    assert _ModelResolver(capture.models).resolve("custom").contract["provider"] == "custom"
    assert reads == ["contract"]


def test_async_callable_loader_result_is_closed_and_rejected() -> None:
    class Loader:
        async def __call__(self):
            return create_model(protocol="openai", model="fixture")
    binding = _ModelResolver({"async": Loader()}).resolve("async")
    with pytest.raises(TypeError):
        _ = binding.contract


@pytest.mark.asyncio
async def test_each_model_resolution_owns_one_immutable_configuration(monkeypatch) -> None:
    import httpx2
    from linktools.ai.model import _openai
    clients, loads = [], []
    configuration = {"api_key": "fixture-secret-1"}
    def client(**kwargs):
        value = httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: pytest.fail("unexpected request")), **kwargs)
        clients.append(value)
        return value
    monkeypatch.setattr(_openai, "DefaultAsyncHttpxClient", client)
    def load():
        loads.append(1)
        return create_model(protocol="openai", model="fixture", api_key=configuration["api_key"])
    features = RuntimeFeatures()
    features.model("assistant", load)
    capture = await features.capture()
    binding = _ModelResolver(capture.models).resolve("assistant")
    contract = dict(binding.contract)
    assert loads == [1] and clients == []
    configuration["api_key"] = "fixture-secret-2"
    for _ in range(2):
        async with binding.materialize() as model:
            assert model.wrapped.provider.client.api_key == "fixture-secret-1"
            assert len(loads) == 1
    assert len(clients) == 2 and all(client.is_closed for client in clients)
    assert dict(binding.contract) == contract
    assert "fixture-secret" not in repr(binding.contract)
    replacement = _ModelResolver(capture.models).resolve("assistant")
    async with replacement.materialize() as model:
        assert model.wrapped.provider.client.api_key == "fixture-secret-2"
    assert loads == [1, 1]
    assert len(clients) == 3 and all(client.is_closed for client in clients)
    assert dict(replacement.contract) == contract


def test_cancelled_loader_does_not_poison_first_success() -> None:
    import asyncio
    calls = []
    def load():
        calls.append(1)
        if len(calls) == 1:
            raise asyncio.CancelledError
        return create_model(protocol="openai", model="fixture")
    binding = _ModelResolver({"assistant": load}).resolve("assistant")
    with pytest.raises(asyncio.CancelledError):
        _ = binding.contract
    assert binding.model_identity == "openai:fixture"
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("session", [False, True, "concurrent"])
async def test_model_loader_refreshes_each_run_and_reuses_within_tool_rounds(session) -> None:
    import asyncio
    from collections import Counter
    from pydantic_ai.models.test import TestModel
    from linktools.ai.capability import ExecutionCapability
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.runtime import Runtime, RuntimeStorage

    loads, materialized, closed, requests = [], [], [], []
    release = asyncio.Event()
    configuration = {"credential": "first"}
    class Model(TestModel):
        def __init__(self, credential):
            super().__init__(call_tools=["echo"], custom_output_text=credential)
            self.credential = credential
        @asynccontextmanager
        async def request_stream(self, *args, **kwargs):
            requests.append(self.credential)
            if session == "concurrent":
                await release.wait()
            async with super().request_stream(*args, **kwargs) as stream:
                yield stream
        async def __aexit__(self, *args):
            closed.append(self.credential)
    class Binding:
        provider = "test"
        model_identity = "test:fixture"
        contract = {"provider": "test", "model_identity": "test:fixture"}
        def __init__(self, credential):
            self.credential = credential
        def materialize(self):
            materialized.append(self.credential)
            return Model(self.credential)
    def load():
        value = configuration["credential"]
        loads.append(value)
        return Binding(value)
    features = RuntimeFeatures()
    features.model("default", load)
    features.agent("default", model="default")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    @features.tool
    def echo(ctx: AgentContext[object]) -> str:
        return "echo"
    async with Runtime.open("per-run-model", storage=RuntimeStorage.in_memory(), features=features) as runtime:
        assert loads == []
        agent = runtime.agents.get()
        target = await agent.create_session("session") if session is True else agent
        if session == "concurrent":
            first = await target.start("run")
            configuration["credential"] = "second"
            second = await target.start("run")
            release.set()
            results = await asyncio.gather(first.wait(timeout_seconds=5), second.wait(timeout_seconds=5))
        else:
            results = []
            for credential in ("first", "second"):
                configuration["credential"] = credential
                results.append(await target.run("run", timeout_seconds=5))
        assert all(result.result.status is ExecutionStatus.SUCCEEDED for result in results)
        assert loads == ["first", "second"]
        assert Counter(materialized) == {"first": 1, "second": 1}
        assert Counter(requests) == {"first": 2, "second": 1 if session is True else 2}
        assert Counter(closed) == {"first": 1, "second": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("change_model", [False, True])
async def test_task_run_refreshes_configuration_under_the_same_registered_model(change_model) -> None:
    from pydantic_ai.models.test import TestModel
    from linktools.ai.core import TaskStatus
    from linktools.ai.runtime import Runtime, RuntimeStorage, AgentTaskInput
    from linktools.ai.task import TaskGraph, TaskNode

    state = {"credential": "first", "model": "fixture"}
    loads, actual = [], []
    class Binding:
        provider = "test"
        def __init__(self):
            self.credential = state["credential"]
            self.model_identity = "test:" + state["model"]
            self.contract = {"provider": "test", "model_identity": self.model_identity}
        def materialize(self):
            actual.append(self.credential)
            return TestModel(custom_output_text="ok")
    def load():
        loads.append(state["credential"])
        return Binding()
    features = RuntimeFeatures()
    features.model("default", load)
    features.agent("default", model="default")
    async with Runtime.open("task-model", features=features, storage=RuntimeStorage.in_memory()) as runtime:
        task = runtime.tasks.from_agent("answer", runtime.agents.get())
        assert loads == ["first"]  # Task declaration freezes the model contract.
        state["credential"] = "second"
        if change_model:
            state["model"] = "different"
        graph = TaskGraph("graph", (TaskNode("answer", task=task, input=AgentTaskInput("hi")),))
        run = await runtime.tasks.bind(task).start(graph, idempotency_key="once")
        result = (await run.wait(timeout_seconds=5)).result
        assert result.status is TaskStatus.SUCCEEDED
        assert loads == ["first", "second"]
        assert actual == ["second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["active", "reserved"])
async def test_recovery_uses_one_fresh_binding_for_validation_and_model(tmp_path, phase) -> None:
    from pydantic_ai.models.function import FunctionModel
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.runtime import Runtime, RuntimeStorage
    from ._runtime_test_helpers import _Models
    from ._runtime_process_test_helpers import _NAMESPACE, _RuntimeProcess, _capabilities, _storage

    database = tmp_path / "recovery.db"
    owner = _RuntimeProcess(database, phase=phase)
    execution_id = owner.initial["execution_id"]
    owner.crash()
    owner.close()
    loads, calls = [], []
    async def model(messages, info):
        calls.append(1)
        yield "recovered"
    def load():
        loads.append(1)
        binding = _Models(FunctionModel(stream_function=model))
        binding.contract = dict(binding.contract, model="updated", api="anthropic")
        return binding
    features = _capabilities()
    features.model("default", load)
    async with Runtime.open(_NAMESPACE, storage=_storage(database, phase), features=features) as runtime:
        before = len(loads)
        if phase == "reserved":
            handle = await runtime.agents.get("default").session("session").start("Wait", idempotency_key="start")
            assert handle.execution_id == execution_id
        else:
            await runtime.executions.recover(execution_id, principal=runtime.default_principal, idempotency_key="recover")
        result = await runtime.executions.wait(execution_id, principal=runtime.default_principal, timeout_seconds=10)
        assert result.result.status is ExecutionStatus.SUCCEEDED
        assert len(loads) - before == 1
        assert calls == [1]
        if phase == "active":
            await runtime.executions.recover(execution_id, principal=runtime.default_principal, idempotency_key="recover")
            assert len(loads) - before == 1
            assert calls == [1]


@pytest.mark.parametrize("protocol", ["openai", "openai-responses", "anthropic"])
def test_registered_model_identity_is_independent_of_configuration(protocol) -> None:
    from linktools.ai.agent import AgentCompiler
    from linktools.ai.spec import AgentSpec

    state = {"binding": create_model(protocol="openai", model="original",
                                     model_settings={"extra_body": {"custom": False}})}
    spec = AgentSpec("agent", model="registered")
    compiler = AgentCompiler(models={"registered": lambda: state["binding"]}, candidates=(), agents={"agent": spec})
    compiled = compiler.compile(spec)
    original = compiler.bind(compiled)
    state["binding"] = create_model(
        protocol=protocol,
        model='replacement',
        model_settings={'max_tokens': 4096, 'extra_body': {'custom': True}},
    )
    current = compiler.bind(compiled)
    assert current.binding_digest == original.binding_digest
    assert current.binding_contract.model_contract != original.binding_contract.model_contract
    assert current.binding_contract == original.binding_contract
    restored = compiler.restore(original.binding_contract)
    assert restored.binding_digest == original.binding_digest
    assert restored.compiled_agent.model.model_identity == state["binding"].model_identity
    assert restored.compiled_agent.model.contract == state["binding"].contract
    # Historical diagnostics remain historical; actual execution uses current config.
    assert restored.binding_contract.model_contract == original.binding_contract.model_contract
    alias_spec = AgentSpec("agent", model="different-id")
    alias_compiler = AgentCompiler(models={"different-id": state["binding"]}, candidates=(), agents={"agent": alias_spec})
    alias = alias_compiler.bind(alias_compiler.compile(alias_spec))
    assert alias.binding_digest != original.binding_digest
    with pytest.raises(AIError) as error:
        alias_compiler.restore(original.binding_contract)
    assert error.value.code == ErrorCode.AGENT_BINDING_UNAVAILABLE
