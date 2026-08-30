#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subagent repository-instruction pin inheritance contracts."""

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    Page,
    Principal,
    TenantAuthorizationPolicy,
)
from linktools.ai.runtime import ExecutionRequest, RuntimeDomain, RuntimeState
from linktools.ai.runtime._execution import CancelEffectOutcome, DefaultExecutionService
from linktools.ai.runtime._object import RuntimeObjectKeyFactory
from linktools.ai.runtime.state import ExecutionRecord, RuntimePayloadRef
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import PayloadPolicy, StoredPayload
from linktools.ai.workspace import RepositoryInstructionDocument, RepositoryInstructions


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


class _Catalog:
    def binding(self, digest: str) -> object:
        return SimpleNamespace(digest=digest, snapshot=_binding(digest))


class _History:
    async def history(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[object]:
        del execution_id, tenant_id, cursor, limit
        return Page((), None)

    async def trace(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[object]:
        del execution_id, tenant_id, cursor, limit
        return Page((), None)

    async def transcript(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[object]:
        del execution_id, tenant_id, cursor, limit
        return Page((), None)


class _Launcher:
    def __init__(self, repository: object) -> None:
        self._repository = repository
        self.launches = 0

    async def prepare_start(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        identity: object,
    ) -> ExecutionRecord:
        del request, identity
        started = replace(execution, status=ExecutionStatus.STARTED)
        await self._repository.compare_and_swap(
            execution.execution_id,
            tenant_id=execution.tenant_id,
            expected_revision=execution.revision,
            next_record=started,
        )
        return started

    async def launch(self, request: ExecutionRequest, execution: ExecutionRecord) -> None:
        del request, execution
        self.launches += 1

    async def cancel(self, execution: object) -> CancelEffectOutcome:
        del execution
        return CancelEffectOutcome.CONFIRMED

    def worker_failure(self, execution_id: str, *, tenant_id: str):
        del execution_id, tenant_id
        return None


class _Resolver:
    def __init__(self, instructions: RepositoryInstructions) -> None:
        self.instructions = instructions
        self.calls: list[str] = []

    async def resolve(self, target: str, *, exclude_sources=frozenset()) -> RepositoryInstructions:
        del exclude_sources
        self.calls.append(target)
        return self.instructions


def _request(key: str) -> ExecutionRequest:
    return ExecutionRequest(
        user_prompt="work",
        user_prompt_codec="text",
        principal=Principal("owner", "tenant"),
        idempotency_key=key,
        memory_scope="scope",
        mode="run",
        planning=False,
        thinking=False,
    )


def _pin(content: str) -> tuple[RepositoryInstructions, RuntimePayloadRef]:
    instructions = RepositoryInstructions(
        (RepositoryInstructionDocument("agents:AGENTS.md", ".", content),)
    )
    return (
        instructions,
        RuntimePayloadRef(StoredPayload.inline_json(instructions.to_payload()), RuntimeDomain.EXECUTION),
    )


def _parent(pin: RuntimePayloadRef | None) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="parent",
        tenant_id="tenant",
        session_id=None,
        binding_digest="a" * 64,
        parent_execution_id=None,
        root_execution_id="parent",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=1,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding("a" * 64),
        repository_instructions=pin,
    )


def _service(
    state: RuntimeState,
    *,
    resolver: _Resolver | None,
    ids: tuple[str, ...],
) -> DefaultExecutionService:
    kwargs: dict[str, object] = {}
    if resolver is not None:
        kwargs.update(
            instruction_resolver=resolver,
            object_key_factory=RuntimeObjectKeyFactory("subagent-instructions"),
            payload_policy=PayloadPolicy(),
        )
    return DefaultExecutionService(
        state.execution,
        state.object_store(RuntimeDomain.EXECUTION),
        TenantAuthorizationPolicy(),
        sessions=state.conversation.sessions,
        catalog=_Catalog(),
        compiler=object(),
        backend=_Launcher(state.execution.executions),
        operation_ids=iter(ids).__next__,
        history_reader=_History(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_new_child_inherits_exact_parent_structured_pin_without_live_resolution() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="subagent-inherit", tenant_id="tenant")
    try:
        _, parent_pin = _pin("parent-v1")
        await state.execution.executions.create(_parent(parent_pin))
        resolver = _Resolver(_pin("live-root")[0])
        service = _service(state, resolver=resolver, ids=("child",))
        handle = await service.start_subagent(
            "a" * 64,
            _request("child-key"),
            parent_execution_id="parent",
            root_execution_id="parent",
        )
        child = await state.execution.executions.get(handle.execution_id, tenant_id="tenant")
        assert child is not None
        assert child.repository_instructions == parent_pin
        assert resolver.calls == []
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_instruction_aware_child_resolves_root_when_parent_pin_is_none() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="subagent-root", tenant_id="tenant")
    try:
        await state.execution.executions.create(_parent(None))
        root, _ = _pin("resolved-root")
        resolver = _Resolver(root)
        service = _service(state, resolver=resolver, ids=("child",))
        handle = await service.start_subagent(
            "a" * 64,
            _request("child-key"),
            parent_execution_id="parent",
            root_execution_id="parent",
        )
        child = await state.execution.executions.get(handle.execution_id, tenant_id="tenant")
        assert child is not None and child.repository_instructions is not None
        assert child.repository_instructions.payload.digest == root.digest
        assert resolver.calls == ["."]
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_parent_pin", [False, True])
async def test_standalone_service_without_resolver_never_assigns_child_pin(with_parent_pin: bool) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace=f"subagent-standalone-{with_parent_pin}", tenant_id="tenant")
    try:
        parent_pin = _pin("parent-v1")[1] if with_parent_pin else None
        await state.execution.executions.create(_parent(parent_pin))
        service = _service(state, resolver=None, ids=("child",))
        handle = await service.start_subagent(
            "a" * 64,
            _request("child-key"),
            parent_execution_id="parent",
            root_execution_id="parent",
        )
        child = await state.execution.executions.get(handle.execution_id, tenant_id="tenant")
        assert child is not None
        assert child.repository_instructions is None
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_subagent_idempotent_replay_keeps_first_persisted_pin_after_live_root_changes() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="subagent-replay", tenant_id="tenant")
    try:
        await state.execution.executions.create(_parent(None))
        first, _ = _pin("first")
        resolver = _Resolver(first)
        service = _service(state, resolver=resolver, ids=("child", "unused"))
        request = _request("child-key")
        first_handle = await service.start_subagent(
            "a" * 64,
            request,
            parent_execution_id="parent",
            root_execution_id="parent",
        )
        first_child = await state.execution.executions.get(first_handle.execution_id, tenant_id="tenant")
        assert first_child is not None and first_child.repository_instructions is not None
        first_pin = first_child.repository_instructions
        resolver.instructions = _pin("changed")[0]
        replay = await service.start_subagent(
            "a" * 64,
            request,
            parent_execution_id="parent",
            root_execution_id="parent",
        )
        replay_child = await state.execution.executions.get(replay.execution_id, tenant_id="tenant")
        assert replay.execution_id == first_handle.execution_id
        assert replay_child is not None and replay_child.repository_instructions == first_pin
        assert resolver.calls == ["."]
    finally:
        await state.close()
