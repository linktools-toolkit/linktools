#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal ownership preserves concurrent dependency retention metadata."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionStatus,
    StopReason,
    UsageMetrics,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    Runtime,
    RuntimeStorage,
    RuntimeStoragePlan,
    RuntimeStorageRoute,
)
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.state._contracts import (
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    ResultRecord,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    RecoveryTerminalHandoff,
    RecoveryTerminalOutcome,
)
from linktools.ai.runtime.state._runtime_commands import RuntimeStateCommands
from .test_history_projection_conformance import _record
from .test_session_admission import _session
from .test_evaluation_consumers import CONTEXT, PRINCIPAL, FixtureModels


def _storage(tmp_path: Path, backend: str, split_session: bool) -> RuntimeStorage:
    if split_session:
        route = (
            RuntimeStorageRoute.memory()
            if backend == "memory"
            else RuntimeStorageRoute.filesystem(tmp_path / "execution")
            if backend == "filesystem"
            else RuntimeStorageRoute.sqlite(tmp_path / "execution.db")
        )
        return RuntimeStorage(
            RuntimeStoragePlan(
                execution=route,
                conversation=RuntimeStorageRoute.filesystem(tmp_path / "conversation"),
            )
        )
    if backend == "memory":
        return RuntimeStorage.in_memory()
    if backend == "filesystem":
        return RuntimeStorage.filesystem(tmp_path / "runtime")
    return RuntimeStorage.sqlite(tmp_path / "runtime.db")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "split_session"),
    (
        ("memory", False),
        ("filesystem", False),
        ("sqlite", False),
        ("filesystem", True),
        ("sqlite", True),
    ),
)
@pytest.mark.parametrize("hold_change", ("acquire", "release"))
async def test_terminal_preserves_concurrent_hold_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    split_session: bool,
    hold_change: str,
) -> None:
    original_success = LocalExecutionBackend._commit_success
    original_body = (
        LocalExecutionBackend._commit_execution_terminal_checkpoint_locked_body
    )
    changed = []
    terminal_results = []

    async def prepare_hold(self, execution, binding, output, usage, agent_run_id):
        if hold_change == "release":
            await self._execution.executions.acquire_dependency_hold(
                execution.execution_id,
                tenant_id=self._tenant_id,
                hold_id="dependent",
            )
        return await original_success(
            self, execution, binding, output, usage, agent_run_id
        )

    async def interleave(self, current, commit, **kwargs):
        if commit.execution.status is ExecutionStatus.SUCCEEDED:
            operation = (
                self._execution.executions.acquire_dependency_hold
                if hold_change == "acquire"
                else self._execution.executions.release_dependency_hold
            )
            await operation(
                current.execution_id,
                tenant_id=self._tenant_id,
                hold_id="dependent",
            )
            observed = await self._execution.executions.get(
                current.execution_id,
                tenant_id=self._tenant_id,
            )
            assert observed is not None
            assert observed.revision == current.revision + 1
            assert (
                replace(
                    observed,
                    revision=current.revision,
                    updated_at=current.updated_at,
                    dependency_hold_ids=current.dependency_hold_ids,
                )
                == current
            )
            changed.append(observed)
            terminal_results.append(commit.result)
        return await original_body(self, current, commit, **kwargs)

    monkeypatch.setattr(LocalExecutionBackend, "_commit_success", prepare_hold)
    monkeypatch.setattr(
        LocalExecutionBackend,
        "_commit_execution_terminal_checkpoint_locked_body",
        interleave,
    )
    group = CapabilityGroup("terminal-metadata")
    group.agent(
        "default", model="default", allow_tools=(), allow_skills=(), allow_subagents=()
    )
    state = _storage(tmp_path, backend, split_session)
    async with Runtime.open(
        "terminal-metadata",
        models=FixtureModels(),
        storage=state,
        context=CONTEXT,
        capabilities=(group,),
    ) as runtime:
        agent = runtime.agents.get()
        if split_session:
            await agent.create_session("session", principal=PRINCIPAL)
            handle = await agent.session("session").start("hello", principal=PRINCIPAL)
        else:
            handle = await agent.start("hello", principal=PRINCIPAL)
        result = await handle.wait(timeout_seconds=20)
        assert changed
        assert result.status is ExecutionStatus.SUCCEEDED
        observed = await state.execution.executions.get(
            handle.execution_id, tenant_id=runtime.tenant_id
        )
        assert observed is not None
        assert observed.dependency_hold_ids == changed[-1].dependency_hold_ids
        assert observed.updated_at >= changed[-1].updated_at
        assert observed.result == terminal_results[-1]
        assert observed.result.usage.model_requests == 1
        assert (
            await state.execution.executions.get_history_seal(
                handle.execution_id, tenant_id=runtime.tenant_id
            )
            is not None
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "split_session"),
    (("memory", False), ("filesystem", False), ("sqlite", False), ("sqlite", True)),
)
@pytest.mark.parametrize(
    "post_commit_change",
    ("hold", "clock", "binding", "result", "event", "status", "seal", "recovery"),
)
async def test_unknown_terminal_commit_verifies_semantics_and_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    split_session: bool,
    post_commit_change: str,
) -> None:
    state = _storage(tmp_path, backend, split_session)
    await state.initialize(namespace="terminal-readback", tenant_id="tenant")
    repository = state.execution.executions
    current = _record(ExecutionStatus.STARTED, 0)
    if split_session:
        current = replace(current, session_id="session")
        await state.conversation.sessions.create(
            replace(_session(), active_execution_id="execution")
        )
    await repository.create_with_history_head(current)
    result = ResultRecord(None, StopReason.ERROR, UsageMetrics(), current.updated_at)
    terminal = replace(
        current,
        status=ExecutionStatus.FAILED,
        error_code=ErrorCode.EXECUTION_FAILED.value,
    )
    commit = ExecutionTerminalCommit(
        current.revision,
        current.event_sequence,
        terminal,
        result,
        ExecutionEventType.EXECUTION_FAILED,
        {"error_code": ErrorCode.EXECUTION_FAILED.value, "safe_error_details": {}},
    )
    source_recovery = RecoveryCheckpoint(
        "execution",
        None,
        RecoveryCheckpointState.HANDOFF,
        0,
        current.created_at,
        current.updated_at,
        handoff_phase=RecoveryHandoffPhase.PREPARED,
        terminal_handoff=RecoveryTerminalHandoff(
            RecoveryTerminalOutcome(
                terminal.status,
                terminal.error_code,
                {},
                result.stop_reason,
                None,
                None,
                result.usage,
                commit.terminal_event_type,
                commit.terminal_event_payload,
                result.created_at,
            ),
            None,
            None,
        ),
    )
    await state.recovery.checkpoints.create(source_recovery)
    target_recovery = replace(
        source_recovery,
        revision=1,
        state=RecoveryCheckpointState.COMPLETED,
        handoff_phase=RecoveryHandoffPhase.COMPLETED,
        terminal_handoff=None,
    )
    commands = RuntimeStateCommands(
        repository,
        namespace="terminal-readback",
        events=state.execution.events,
        conversation=state.conversation.sessions,
        recovery=state.recovery.checkpoints,
        background_tasks=set(),
    )
    group = repository.state_store.storage_group
    original = group.mutate
    written = []

    async def lose_response(stores, callback):
        value = await original(stores, callback)
        if isinstance(value, ExecutionTerminalCommitResult):
            written.append(value)
            if post_commit_change == "hold":
                await repository.acquire_dependency_hold(
                    "execution", tenant_id="tenant", hold_id="after"
                )
                await repository.release_dependency_hold(
                    "execution", tenant_id="tenant", hold_id="after"
                )
            elif post_commit_change == "recovery":
                await state.recovery.checkpoints.compare_and_swap(
                    "execution",
                    tenant_id="tenant",
                    expected_revision=1,
                    next_record=replace(source_recovery, revision=2),
                )
            elif post_commit_change == "seal":

                async def remove_seal(transaction):
                    key = repository._key("execution_history_seal", "execution")
                    stored = await transaction.get_record(key)
                    assert stored is not None
                    await transaction.delete_record(
                        key, expected_storage_version=stored.storage_version
                    )

                await repository.state_store.mutate(remove_seal)
            else:
                changed = (
                    replace(
                        value.execution,
                        updated_at=value.execution.updated_at - timedelta(seconds=1),
                    )
                    if post_commit_change == "clock"
                    else replace(
                        value.execution,
                        binding=replace(
                            value.execution.binding, model_contract={"model": "changed"}
                        ),
                    )
                    if post_commit_change == "binding"
                    else replace(
                        value.execution,
                        result=replace(
                            result, created_at=result.created_at + timedelta(seconds=1)
                        ),
                    )
                    if post_commit_change == "result"
                    else replace(value.execution, status=ExecutionStatus.CANCELLED)
                    if post_commit_change == "status"
                    else replace(
                        value.execution,
                        event_sequence=value.execution.event_sequence + 1,
                    )
                )
                await repository.compare_and_swap(
                    "execution",
                    tenant_id="tenant",
                    expected_revision=value.execution.revision,
                    next_record=replace(changed, revision=value.execution.revision + 1),
                )
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        return value

    await repository.acquire_dependency_hold(
        "execution", tenant_id="tenant", hold_id="before"
    )
    monkeypatch.setattr(group, "mutate", lose_response)
    try:
        if post_commit_change in {"hold", "clock"}:
            observed = await commands.commit_terminal_checkpoint(
                commit,
                expected_execution=current,
                session_id=current.session_id,
                recovery_checkpoint=target_recovery,
            )
            assert observed.execution.status is ExecutionStatus.FAILED
            assert observed.result == result
            assert observed.execution.revision == written[0].execution.revision + (
                2 if post_commit_change == "hold" else 1
            )
            assert observed.execution.dependency_hold_ids == ("before",)
        else:
            with pytest.raises(AIError) as raised:
                await commands.commit_terminal_checkpoint(
                    commit,
                    expected_execution=current,
                    session_id=current.session_id,
                    recovery_checkpoint=target_recovery,
                )
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert len(written) == 1
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ("binding", "event", "status", "sealed", "recovery"))
async def test_terminal_rebase_rejects_semantic_changes_without_writes(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="terminal-conflict", tenant_id="tenant")
    repository = state.execution.executions
    current = _record(ExecutionStatus.STARTED, 0)
    await repository.create_with_history_head(current)
    result = ResultRecord(None, StopReason.ERROR, UsageMetrics(), current.updated_at)
    terminal = replace(
        current,
        status=ExecutionStatus.FAILED,
        error_code=ErrorCode.EXECUTION_FAILED.value,
    )
    commit = ExecutionTerminalCommit(
        current.revision,
        current.event_sequence,
        terminal,
        result,
        ExecutionEventType.EXECUTION_FAILED,
        {"error_code": ErrorCode.EXECUTION_FAILED.value, "safe_error_details": {}},
    )
    commands = RuntimeStateCommands(
        repository,
        namespace="terminal-conflict",
        events=state.execution.events,
        recovery=state.recovery.checkpoints,
        background_tasks=set(),
    )
    recovery = None
    if drift in {"sealed", "recovery"}:
        # A terminal seal without its terminal execution must never be treated as
        # a benign revision conflict, even when the execution is unchanged.
        if drift == "sealed":
            from linktools.ai.runtime.state._execution_commands import (
                _execution_history_seal,
            )

            seal = _execution_history_seal(
                commit,
                audit_events=(),
                projections=(),
                current_run=None,
                current_events=(),
                current_batch=None,
            )
            await repository.state_store.mutate(
                lambda tx: repository.put_history_seal_in_transaction(tx, seal)
            )
        else:
            from linktools.ai.runtime.state._contracts import (
                RecoveryCheckpoint,
                RecoveryCheckpointState,
                RecoveryHandoffPhase,
            )

            recovery = RecoveryCheckpoint(
                "execution",
                None,
                RecoveryCheckpointState.COMPLETED,
                1,
                current.created_at,
                current.updated_at,
                handoff_phase=RecoveryHandoffPhase.COMPLETED,
            )
            await state.recovery.checkpoints.create(recovery)

        async def abort_mutation(stores, callback):
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(
            repository.state_store.storage_group, "mutate", abort_mutation
        )
    else:
        changed = (
            replace(
                current,
                binding=replace(current.binding, model_contract={"model": "changed"}),
            )
            if drift == "binding"
            else replace(current, event_sequence=current.event_sequence + 1)
            if drift == "event"
            else replace(current, status=ExecutionStatus.RECOVERY_REQUIRED)
        )
        await repository.compare_and_swap(
            "execution",
            tenant_id="tenant",
            expected_revision=current.revision,
            next_record=replace(changed, revision=current.revision + 1),
        )
    before = await repository.get("execution", tenant_id="tenant")
    try:
        with pytest.raises(AIError) as raised:
            await commands.commit_terminal_checkpoint(
                commit, expected_execution=current, recovery_checkpoint=recovery
            )
        assert raised.value.code is (
            ErrorCode.STORAGE_INTEGRITY_ERROR
            if drift in {"sealed", "recovery"}
            else ErrorCode.EXECUTION_RESULT_CONFLICT
        )
        assert await repository.get("execution", tenant_id="tenant") == before
        events = await state.execution.events.list(
            "execution", tenant_id="tenant", after_sequence=0, limit=10
        )
        assert not events.items
    finally:
        await state.close()

