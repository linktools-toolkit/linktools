#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem state, lifecycle, and subagent precedence evidence."""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.core import Principal, SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState, RuntimeStatePlan, RuntimeStateRoute
from linktools.ai.runtime._subagent import SubagentDispatcher
from linktools.ai.runtime.state import _materializer as materializer
from linktools.ai.runtime.state._contracts import SessionRecord
from linktools.ai.runtime.state._filesystem import _FilesystemDomainBackend
from linktools.ai.runtime.state._plan import RuntimeDomain


def _scope(root: Path, namespace: str, tenant_id: str) -> Path:
    return root / hashlib.sha256(namespace.encode()).hexdigest() / hashlib.sha256(tenant_id.encode()).hexdigest()


def _plan(root: Path) -> RuntimeStatePlan:
    return RuntimeStatePlan(conversation=RuntimeStateRoute.filesystem(root))


async def _create_session(state: RuntimeState, tenant_id: str, session_id: str) -> None:
    now = datetime.now(timezone.utc)
    await state.conversation.sessions.create(
        SessionRecord(
            session_id=session_id,
            tenant_id=tenant_id,
            owner_principal_id="owner",
            binding_digest="binding",
            status=SessionStatus.OPEN,
            revision=0,
            resource_generation=0,
            cwd=None,
            metadata={},
            created_at=now,
            updated_at=now,
            closed_at=None,
        )
    )


@pytest.mark.asyncio
async def test_filesystem_state_isolated_by_namespace_and_tenant(tmp_path: Path) -> None:
    route_root = tmp_path / "route"
    state = RuntimeState.from_plan(_plan(route_root))
    await state.initialize(namespace="namespace-a", tenant_id="tenant-a")
    await _create_session(state, "tenant-a", "session-a")
    await state.close()

    other_namespace = RuntimeState.from_plan(_plan(route_root))
    await other_namespace.initialize(namespace="namespace-b", tenant_id="tenant-a")
    assert await other_namespace.conversation.sessions.get("session-a", tenant_id="tenant-a") is None
    await other_namespace.close()

    other_tenant = RuntimeState.from_plan(_plan(route_root))
    await other_tenant.initialize(namespace="namespace-a", tenant_id="tenant-b")
    assert await other_tenant.conversation.sessions.get("session-a", tenant_id="tenant-b") is None
    await other_tenant.close()

    scope = _scope(route_root, "namespace-a", "tenant-a")
    assert (scope / "manifest.json").is_file()
    assert (scope / "records.json").is_file()
    assert (scope / "steps").is_dir()
    assert scope == route_root / scope.relative_to(route_root)


@pytest.mark.asyncio
async def test_filesystem_components_are_closed_once_in_normal_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized: list[int] = []
    closed: list[int] = []
    initialize = materializer._initialize_components
    close_once = materializer._close_once

    async def record_initialize(components: tuple[object, ...], cleanup: list[object]) -> None:
        initialized.extend(id(component) for component in components)
        await initialize(components, cleanup)

    def record_close(close_method: object) -> object:
        owner = getattr(close_method, "__self__", None)
        action = close_once(close_method)

        async def close() -> None:
            if owner is not None:
                closed.append(id(owner))
            await action()

        return close

    monkeypatch.setattr(materializer, "_initialize_components", record_initialize)
    monkeypatch.setattr(materializer, "_close_once", record_close)
    value = await materializer.materialize_runtime_state(
        _plan(tmp_path / "route"),
        namespace="n",
        tenant_id="t",
        object_store=None,
    )
    for action in value.close_actions:
        await action()

    assert closed == list(reversed(initialized))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("manifest", "records", "expected"),
    (
        (
            {"format": "old", "generation": 1, "namespace": "n", "tenant_id": "t", "domain": "conversation"},
            None,
            ErrorCode.STORAGE_VERSION_UNSUPPORTED,
        ),
        (
            {
                "format": "linktools-ai-runtime-state",
                "generation": 2,
                "namespace": "n",
                "tenant_id": "t",
                "domain": "conversation",
            },
            None,
            ErrorCode.STORAGE_VERSION_UNSUPPORTED,
        ),
        (
            {
                "format": "linktools-ai-runtime-state",
                "generation": 1,
                "namespace": "wrong",
                "tenant_id": "t",
                "domain": "conversation",
            },
            None,
            ErrorCode.STORAGE_INTEGRITY_ERROR,
        ),
        (
            {
                "format": "linktools-ai-runtime-state",
                "generation": 1,
                "namespace": "n",
                "tenant_id": "wrong",
                "domain": "conversation",
            },
            None,
            ErrorCode.STORAGE_INTEGRITY_ERROR,
        ),
        (
            {
                "format": "linktools-ai-runtime-state",
                "generation": 1,
                "namespace": "n",
                "tenant_id": "t",
                "domain": "execution",
            },
            None,
            ErrorCode.STORAGE_INTEGRITY_ERROR,
        ),
        (
            {
                "format": "linktools-ai-runtime-state",
                "generation": 1,
                "namespace": "n",
                "tenant_id": "t",
                "domain": "conversation",
            },
            {"sessions": "malformed"},
            ErrorCode.STORAGE_INTEGRITY_ERROR,
        ),
    ),
)
async def test_filesystem_manifest_and_records_integrity(
    tmp_path: Path,
    manifest: dict[str, object],
    records: dict[str, object] | None,
    expected: ErrorCode,
) -> None:
    route_root = tmp_path / "route"
    scope = _scope(route_root, "n", "t")
    scope.mkdir(parents=True)
    (scope / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if records is not None:
        (scope / "records.json").write_text(json.dumps(records), encoding="utf-8")

    state = RuntimeState.from_plan(_plan(route_root))
    with pytest.raises(AIError) as error:
        await state.initialize(namespace="n", tenant_id="t")
    assert error.value.code is expected


@pytest.mark.asyncio
async def test_filesystem_missing_manifest_does_not_adopt_existing_records(tmp_path: Path) -> None:
    route_root = tmp_path / "route"
    scope = _scope(route_root, "n", "t")
    scope.mkdir(parents=True)
    (scope / "records.json").write_text(json.dumps({"sessions": []}), encoding="utf-8")
    state = RuntimeState.from_plan(_plan(route_root))
    with pytest.raises(AIError) as error:
        await state.initialize(namespace="n", tenant_id="t")
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_filesystem_release_retries_after_writer_lock_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FilesystemDomainBackend(
        tmp_path / "route",
        namespace="n",
        tenant_id="t",
        domain=RuntimeDomain.CONVERSATION,
    )
    await backend.prepare()
    release = backend.writer_lock.release
    calls = 0

    async def fail_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("release failed")
        await release()

    monkeypatch.setattr(backend.writer_lock, "release", fail_once)
    with pytest.raises(RuntimeError, match="release failed"):
        await backend.release()
    assert backend._released is False
    await backend.release()
    assert backend._released is True
    assert calls == 2


@pytest.mark.asyncio
async def test_filesystem_prepare_preserves_primary_when_release_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FilesystemDomainBackend(
        tmp_path / "route",
        namespace="n",
        tenant_id="t",
        domain=RuntimeDomain.CONVERSATION,
    )
    primary = RuntimeError("load failed")

    def load() -> None:
        raise primary

    monkeypatch.setattr(backend, "_load", load)

    async def release() -> None:
        raise RuntimeError("release failed")

    monkeypatch.setattr(backend.writer_lock, "release", release)
    with pytest.raises(RuntimeError) as error:
        await backend.prepare()
    assert error.value is primary


class _Compiler:
    async def compile_subagent(self, *, agent_id: str) -> SimpleNamespace:
        return SimpleNamespace(digest=agent_id)


class _Execution:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def start_subagent(self, *args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(execution_id="child")

    async def wait(self, execution_id: str, *, principal: Principal) -> None:
        raise self.error


def _dispatch(dispatcher: SubagentDispatcher) -> object:
    return dispatcher.dispatch(
        parent_execution_id="parent",
        root_execution_id="root",
        memory_scope=None,
        principal=Principal("owner", "tenant"),
        agent_id="agent",
        user_prompt="task",
    )


@pytest.mark.asyncio
async def test_subagent_cleanup_cancellation_does_not_replace_primary() -> None:
    primary = RuntimeError("wait failed")
    started = asyncio.Event()
    release = asyncio.Event()

    class Dispatcher(SubagentDispatcher):
        async def cancel_child(self, *args: object, **kwargs: object) -> None:
            started.set()
            await release.wait()

    dispatcher = Dispatcher(_Compiler(), {}, _Execution(primary))
    task = asyncio.create_task(_dispatch(dispatcher))
    await started.wait()
    task.cancel()
    release.set()
    with pytest.raises(RuntimeError, match="wait failed") as error:
        await task
    assert error.value is primary


@pytest.mark.asyncio
async def test_subagent_cleanup_failure_is_recovery_error_from_primary() -> None:
    primary = RuntimeError("wait failed")

    class Dispatcher(SubagentDispatcher):
        async def cancel_child(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("cleanup failed")

    dispatcher = Dispatcher(_Compiler(), {}, _Execution(primary))
    with pytest.raises(AIError) as error:
        await _dispatch(dispatcher)
    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert error.value.__cause__ is primary
