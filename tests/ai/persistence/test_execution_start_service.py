#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Execution service start-claim race coverage."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionStatus,
    Page,
    Principal,
    TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import ExecutionRequest, RuntimeDomain, RuntimeState
from linktools.ai.runtime._execution import CancelEffectOutcome, DefaultExecutionService
from linktools.ai.runtime.state import ExecutionRecord
from linktools.ai.spec import AgentSpec
from sqlalchemy.ext.asyncio import create_async_engine


def _binding(digest: str) -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", model="model"),
        model={"route_id": "model", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
        binding_digest=digest,
    )


class _DefinitionCatalog:
    def binding(self, digest: str) -> object:
        return SimpleNamespace(digest=digest, snapshot=_binding(digest))


class _History:
    async def history(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> Page[object]:
        del execution_id, tenant_id, cursor, limit
        return Page((), None)

    async def trace(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> Page[object]:
        del execution_id, tenant_id, cursor, limit
        return Page((), None)

    async def transcript(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> Page[object]:
        del execution_id, tenant_id, cursor, limit
        return Page((), None)


class _Launcher:
    def __init__(self, repository: object | None = None) -> None:
        self.calls = 0
        self._started: set[str] = set()
        self._repository = repository

    async def prepare_start(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        identity: object,
    ) -> ExecutionRecord:
        del request, identity
        started = replace(execution, status=ExecutionStatus.STARTED)
        repository = self._repository
        if repository is not None:
            await repository.compare_and_swap(
                execution.execution_id,
                tenant_id=execution.tenant_id,
                expected_revision=execution.revision,
                next_record=started,
            )
        return started

    async def launch(self, request: ExecutionRequest, execution: ExecutionRecord) -> None:
        del request
        if execution.execution_id in self._started:
            return
        self._started.add(execution.execution_id)
        self.calls += 1
        await asyncio.sleep(0.01)

    async def cancel(self, execution: object) -> CancelEffectOutcome:
        del execution
        return CancelEffectOutcome.CONFIRMED

    def worker_failure(self, execution_id: str, *, tenant_id: str) -> AIError | None:
        del execution_id, tenant_id
        return None


def _request(
    prompt: str,
    principal: Principal,
    idempotency_key: str,
    *,
    memory_scope: str | None = None,
) -> ExecutionRequest:
    return ExecutionRequest(
        user_prompt=prompt,
        user_prompt_codec="text",
        principal=principal,
        idempotency_key=idempotency_key,
        memory_scope=memory_scope,
        mode="run",
        planning=False,
        thinking=False,
    )


def _service(
    state: RuntimeState,
    *,
    backend: _Launcher | None = None,
    operation_ids: object | None = None,
) -> DefaultExecutionService:
    kwargs: dict[str, object] = {}
    if operation_ids is not None:
        kwargs["operation_ids"] = operation_ids
    return DefaultExecutionService(
        state.execution,
        state.object_store(RuntimeDomain.EXECUTION),
        TenantAuthorizationPolicy(),
        sessions=state.conversation.sessions,
        catalog=_DefinitionCatalog(),
        compiler=object(),
        backend=backend,
        history_reader=_History(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_execution_start_claim_has_one_launcher_winner() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="service-start", tenant_id="tenant")
    try:
        launcher = _Launcher(state.execution.executions)
        service = _service(
            state,
            backend=launcher,
            operation_ids=iter(("execution-a", "execution-b")).__next__,
        )
        request = _request(
            "hello",
            Principal("owner", "tenant"),
            "same",
            memory_scope="test",
        )
        first, second = await asyncio.gather(
            service.run("a" * 64, request),
            service.run("a" * 64, request),
        )
        assert first.execution_id == second.execution_id
        assert launcher.calls == 1
        started = await state.execution.executions.get(first.execution_id, tenant_id="tenant")
        assert started is not None
        assert started.agent_run_sequence == 0
        first_attempt = await state.execution.executions.claim_next_agent_run(
            first.execution_id,
            tenant_id="tenant",
            expected_revision=started.revision,
            expected_agent_run_sequence=0,
        )
        assert first_attempt.agent_run_sequence == 1
        second_attempt = await state.execution.executions.claim_next_agent_run(
            first.execution_id,
            tenant_id="tenant",
            expected_revision=first_attempt.revision,
            expected_agent_run_sequence=1,
        )
        assert second_attempt.agent_run_sequence == 2
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_sql_execution_start_keeps_attempt_sequence_zero(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}")
    await provision_database(engine)
    state = RuntimeState.sql(engine)
    await state.initialize(namespace="sql-start", tenant_id="tenant")
    try:
        service = _service(state, backend=_Launcher(state.execution.executions))
        handle = await service.run(
            "b" * 64,
            _request("hello", Principal("owner", "tenant"), "sql-start-key"),
        )
        started = await state.execution.executions.get(handle.execution_id, tenant_id="tenant")
        assert started is not None
        assert started.agent_run_sequence == 0
    finally:
        await state.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_filesystem_execution_start_keeps_attempt_sequence_zero(tmp_path) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="filesystem-start", tenant_id="tenant")
    try:
        service = _service(state, backend=_Launcher(state.execution.executions))
        handle = await service.run(
            "c" * 64,
            _request(
                "hello",
                Principal("owner", "tenant"),
                "filesystem-start-key",
            ),
        )
        started = await state.execution.executions.get(
            handle.execution_id,
            tenant_id="tenant",
        )
        assert started is not None
        assert started.agent_run_sequence == 0
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_terminal_verifier_can_be_bound_once() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="terminal-verifier", tenant_id="tenant")
    try:
        service = _service(state)

        with pytest.raises(ValueError):
            service.bind_terminal_verifier(None)

        async def verifier(
            execution: ExecutionRecord,
            status: ExecutionStatus,
            required_step_run_id: str | None,
        ) -> None:
            del execution, status, required_step_run_id

        service.bind_terminal_verifier(verifier)
        with pytest.raises(RuntimeError):
            service.bind_terminal_verifier(verifier)
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_launch_missing_record_is_storage_integrity_failure() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="launch-integrity", tenant_id="tenant")
    try:
        service = _service(state, backend=_Launcher(state.execution.executions))

        with pytest.raises(AIError) as error:
            await service._launch_started(
                _request("hello", Principal("owner", "tenant"), "launch-key"),
                SimpleNamespace(execution_id="missing", tenant_id="tenant"),
                scope="scope",
                idempotency_key_digest="digest",
            )

        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_execution_memory_scope_can_be_disabled_but_not_blank() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="memory-namespace-validation", tenant_id="tenant")
    try:
        service = _service(state, backend=_Launcher(state.execution.executions))
        principal = Principal("owner", "tenant")
        handle = await service.run(
            "a" * 64,
            _request("without memory", principal, "without-memory"),
        )
        execution = await state.execution.executions.get(
            handle.execution_id,
            tenant_id=principal.tenant_id,
        )
        assert execution is not None
        assert execution.memory_scope is None

        for value in ("", "  "):
            with pytest.raises(AIError) as error:
                await service.run(
                    "a" * 64,
                    _request(
                        "invalid memory",
                        principal,
                        f"invalid-{len(value)}",
                        memory_scope=value,
                    ),
                )
            assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    finally:
        await state.close()
