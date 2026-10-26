#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retention cleanup follows the committed closure without redundant owner reads."""

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import event

from linktools.ai.persistence import Persistence
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.persistence._retention import RuntimeRetentionController
from linktools.ai.persistence.state import active_state_scope

from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution as _projection_execution

pytestmark = pytest.mark.asyncio
_NAMESPACE = "retention-io"


def _execution():
    return replace(_projection_execution(), conversation=False)


def _controller(repository, calls):
    controller = object.__new__(RuntimeRetentionController)
    controller._execution = SimpleNamespace(executions=repository)
    controller._namespace = _NAMESPACE
    controller._transient_domains = frozenset()
    controller._objects = SimpleNamespace()
    async def session_release(session_id, **kwargs):
        assert active_state_scope() is None
        record = await repository.get(kwargs["execution_id"], tenant_id=kwargs["tenant_id"])
        assert record.retention_closed and record.session_id == session_id
        calls.append(("session", session_id))
    async def staging_release(**kwargs):
        assert active_state_scope() is None
        record = await repository.get(kwargs["execution_id"], tenant_id="tenant")
        assert record.retention_closed
        assert len(kwargs["candidate_agent_run_ids"]) == record.agent_run_seq
        calls.append(("staging", kwargs["execution_id"]))
    controller._conversation = SimpleNamespace(sessions=SimpleNamespace(release_execution=session_release))
    controller._run_store = SimpleNamespace(release_staging_many=staging_release)
    return controller


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("condition", ("terminal", "closed", "missing", "running", "hold"))
async def test_retention_cleanup_observes_committed_closure(
    tmp_path: Path, backend: str, condition: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        repository = state.execution.executions
        if condition != "missing":
            await repository.create_with_history_head(replace(
                _execution(), session_id="session", agent_run_seq=2,
                status=_execution().status if condition == "running" else ExecutionStatus.SUCCEEDED,
                retention_closed=condition == "closed",
                dependency_hold_ids=("hold",) if condition == "hold" else (),
            ))
        calls = []
        released = await _controller(repository, calls).release_execution_handoff("execution", tenant_id="tenant")
        assert released == (condition not in ("running", "hold"))
        assert calls == ([] if condition in ("missing", "running", "hold") else [
            ("session", "session"), ("staging", "execution"),
        ])
    finally:
        await state.close()


@pytest.mark.parametrize("failure", ("tenant", "cas", "read", "cancel"))
async def test_retention_failure_does_not_start_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        repository = state.execution.executions
        await repository.create_with_history_head(replace(_execution(), status=ExecutionStatus.SUCCEEDED))
        calls = []
        controller = _controller(repository, calls)
        original_get = repository.get
        if failure in ("read", "cancel"):
            async def fail(*args, **kwargs):
                if failure == "cancel":
                    raise asyncio.CancelledError()
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            monkeypatch.setattr(repository, "get", fail)
        elif failure == "cas":
            from linktools.ai.persistence._sql import _SqlTransaction
            async def fail(*args, **kwargs):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            monkeypatch.setattr(_SqlTransaction, "replace_record", fail)
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else AIError):
            await controller.release_execution_handoff(
                "execution", tenant_id="other" if failure == "tenant" else "tenant",
            )
        assert calls == []
        assert not (await original_get("execution", tenant_id="tenant")).retention_closed
    finally:
        await state.close()


async def test_builtin_retention_avoids_owner_reread_and_custom_owner_keeps_fallback(tmp_path: Path) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        repository = state.execution.executions
        await repository.create_with_history_head(replace(
            _execution(), status=ExecutionStatus.SUCCEEDED, retention_closed=True,
        ))
        class CustomRepository(ExecutionRepositoryImpl):
            pass
        custom = CustomRepository(repository.state_store, namespace=_NAMESPACE, tenant_id="tenant")
        engine = repository.state_store.storage_group.context.engine.sync_engine
        totals = []
        for owner in (custom, repository):
            statements = []
            def capture(_connection, _cursor, statement, _parameters, _context, _many):
                statements.append(statement.split()[0].upper())
            calls = []
            controller = _controller(owner, calls)
            async def staging_release(**kwargs):
                assert active_state_scope() is None
                calls.append(kwargs["execution_id"])
            controller._run_store = SimpleNamespace(release_staging_many=staging_release)
            event.listen(engine, "before_cursor_execute", capture)
            try:
                assert await controller.release_execution_handoff("execution", tenant_id="tenant")
            finally:
                event.remove(engine, "before_cursor_execute", capture)
            assert calls == ["execution"]
            assert not set(statements) & {"INSERT", "UPDATE", "DELETE"}
            totals.append(statements.count("SELECT"))
        assert totals[1] < totals[0]
    finally:
        await state.close()


@pytest.mark.parametrize("failure", ("unknown", "cancel_after_commit"))
async def test_retention_unknown_commit_does_not_clean_until_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        repository = state.execution.executions
        await repository.create_with_history_head(replace(_execution(), status=ExecutionStatus.SUCCEEDED))
        original = repository.state_store.mutate
        async def lose_ack(callback):
            outer = active_state_scope() is None
            result = await original(callback)
            if outer:
                if failure == "cancel_after_commit":
                    raise asyncio.CancelledError()
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result
        calls = []
        controller = _controller(repository, calls)
        with monkeypatch.context() as patch:
            patch.setattr(repository.state_store, "mutate", lose_ack)
            with pytest.raises(asyncio.CancelledError if failure == "cancel_after_commit" else AIError):
                await controller.release_execution_handoff("execution", tenant_id="tenant")
        assert calls == []
        assert (await repository.get("execution", tenant_id="tenant")).retention_closed
        assert await controller.release_execution_handoff("execution", tenant_id="tenant")
        assert calls == [("staging", "execution")]
    finally:
        await state.close()


@pytest.mark.parametrize("conversation,closed,held", (
    (False, False, False), (False, True, False), (False, False, True),
    (True, False, False), (True, True, False), (True, False, True),
))
async def test_terminal_policy_hint_still_requires_source_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    conversation: bool, closed: bool, held: bool,
) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        repository = state.execution.executions
        await repository.create_with_history_head(replace(
            _execution(), session_id=None, status=ExecutionStatus.SUCCEEDED,
            conversation=conversation, retention_closed=closed,
            dependency_hold_ids=("hold",) if held else (),
        ))
        calls = []
        controller = _controller(repository, calls)
        original = repository.state_store.mutate
        mutations = []

        async def mutate(callback):
            mutations.append(callback)
            return await original(callback)

        monkeypatch.setattr(repository.state_store, "mutate", mutate)
        released = await controller.release_execution_handoff(
            "execution", tenant_id="tenant", terminal_commit=True,
        )
        assert released is not held
        assert bool(mutations) is not (conversation and closed)
        assert calls == ([] if held else [("staging", "execution")])
        current = await repository.get("execution", tenant_id="tenant")
        assert current is not None
        assert current.retention_closed is not held
    finally:
        await state.close()
