#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime body and cleanup error precedence coverage."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import RuntimeContext, RuntimeHistory, RuntimeState
from linktools.ai.runtime._runtime_service import _open_runtime
from linktools.ai.workspace import Workspace


class _Components:
    metric_control = None
    catalog = object()
    compiler = object()
    execution = object()
    session = object()
    graph = object()
    evaluation = object()
    approval = object()
    external = object()
    event = object()
    artifact = object()
    task_node_runtime = None
    tree_streamer = None

    def __init__(self, close_callback) -> None:
        self.close_callback = close_callback


class _CaptureLogger:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str, *args: object) -> None:
        self.errors.append(message % args)

    def info(self, _message: str, *_args: object) -> None:
        return None


async def _compose(_workspace: Workspace, **_kwargs: object) -> _Components:
    async def close() -> None:
        raise AIError(
            ErrorCode.STORAGE_RECOVERY_REQUIRED,
            "cleanup secret should not replace body",
        )

    return _Components(close)


async def _compose_with_successful_close(
    _workspace: Workspace,
    **_kwargs: object,
) -> _Components:
    async def close() -> None:
        return None

    return _Components(close)


def _workspace(tmp_path: Path) -> Workspace:
    return Workspace.load(tmp_path, workspace_id="lifecycle")


@pytest.mark.asyncio
async def test_runtime_body_error_wins_over_cleanup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._factory as factory

    monkeypatch.setattr(factory, "compose_runtime_components", _compose)
    with pytest.raises(ValueError, match="body secret"):
        async with _open_runtime(
            _workspace(tmp_path),
            context=RuntimeContext(None),
            models=None,
            state=None,
            capabilities=(),
            metrics=None,
        ):
            raise ValueError("body secret")


@pytest.mark.asyncio
async def test_runtime_body_error_wins_when_close_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._factory as factory

    monkeypatch.setattr(
        factory,
        "compose_runtime_components",
        _compose_with_successful_close,
    )
    with pytest.raises(ValueError, match="body secret"):
        async with _open_runtime(
            _workspace(tmp_path),
            context=RuntimeContext(None),
            models=None,
            state=None,
            capabilities=(),
            metrics=None,
        ):
            raise ValueError("body secret")


@pytest.mark.asyncio
async def test_runtime_close_error_still_propagates_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._factory as factory

    monkeypatch.setattr(factory, "compose_runtime_components", _compose)
    with pytest.raises(AIError) as raised:
        async with _open_runtime(
            _workspace(tmp_path),
            context=RuntimeContext(None),
            models=None,
            state=None,
            capabilities=(),
            metrics=None,
        ):
            pass
    assert raised.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED


@pytest.mark.asyncio
async def test_runtime_construction_error_wins_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._factory as factory
    import linktools.ai.runtime._runtime_service as runtime_service

    def fail_runtime(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("construction secret")

    monkeypatch.setattr(factory, "compose_runtime_components", _compose)
    monkeypatch.setattr(runtime_service, "Runtime", fail_runtime)
    with pytest.raises(RuntimeError, match="construction secret"):
        async with _open_runtime(
            _workspace(tmp_path),
            context=RuntimeContext(None),
            models=None,
            state=None,
            capabilities=(),
            metrics=None,
        ):
            pass


@pytest.mark.asyncio
async def test_runtime_cancellation_wins_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._factory as factory

    monkeypatch.setattr(factory, "compose_runtime_components", _compose)
    with pytest.raises(asyncio.CancelledError):
        async with _open_runtime(
            _workspace(tmp_path),
            context=RuntimeContext(None),
            models=None,
            state=None,
            capabilities=(),
            metrics=None,
        ):
            raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_secondary_cleanup_log_excludes_business_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._factory as factory
    import linktools.ai.runtime._runtime_service as runtime_service

    logger = _CaptureLogger()
    monkeypatch.setattr(factory, "compose_runtime_components", _compose)
    monkeypatch.setattr(runtime_service, "_logger", logger)
    with pytest.raises(ValueError, match="body secret"):
        async with _open_runtime(
            _workspace(tmp_path),
            context=RuntimeContext(None),
            models=None,
            state=None,
            capabilities=(),
            metrics=None,
        ):
            raise ValueError("body secret")

    assert logger.errors
    rendered = "\n".join(logger.errors)
    assert "runtime.body" in rendered
    assert "body secret" not in rendered
    assert "cleanup secret" not in rendered


@pytest.mark.asyncio
async def test_runtime_history_body_error_wins_over_state_close_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    original_close = RuntimeState.close

    async def close_with_failure(_state: RuntimeState) -> None:
        raise RuntimeError("history cleanup secret")

    monkeypatch.setattr(RuntimeState, "close", close_with_failure)
    try:
        with pytest.raises(ValueError, match="history body secret"):
            async with RuntimeHistory.open(_workspace(tmp_path), state=state):
                raise ValueError("history body secret")
    finally:
        await original_close(state)


@pytest.mark.asyncio
async def test_runtime_history_body_error_wins_when_close_succeeds(
    tmp_path: Path,
) -> None:
    state = RuntimeState.in_memory()

    with pytest.raises(ValueError, match="history body secret"):
        async with RuntimeHistory.open(_workspace(tmp_path), state=state):
            raise ValueError("history body secret")

    assert state.ready is False


@pytest.mark.asyncio
async def test_runtime_history_close_error_still_propagates_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    original_close = RuntimeState.close

    async def close_with_failure(_state: RuntimeState) -> None:
        raise RuntimeError("history cleanup")

    monkeypatch.setattr(RuntimeState, "close", close_with_failure)
    try:
        with pytest.raises(RuntimeError, match="history cleanup"):
            async with RuntimeHistory.open(_workspace(tmp_path), state=state):
                pass
    finally:
        await original_close(state)


@pytest.mark.asyncio
async def test_runtime_history_cancellation_wins_when_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    original_close = RuntimeState.close

    async def close_with_failure(_state: RuntimeState) -> None:
        raise RuntimeError("history cleanup")

    monkeypatch.setattr(RuntimeState, "close", close_with_failure)
    try:
        with pytest.raises(asyncio.CancelledError):
            async with RuntimeHistory.open(_workspace(tmp_path), state=state):
                raise asyncio.CancelledError
    finally:
        await original_close(state)


@pytest.mark.asyncio
async def test_compose_cleans_state_when_build_arguments_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._factory as factory

    state = RuntimeState.in_memory()
    close_calls = 0
    original_close = RuntimeState.close

    async def count_close(current: RuntimeState) -> None:
        nonlocal close_calls
        close_calls += 1
        await original_close(current)

    def fail_history_reader(
        _workspace: Workspace,
        _state: RuntimeState,
    ) -> None:
        raise RuntimeError("history reader construction failed")

    monkeypatch.setattr(RuntimeState, "close", count_close)
    monkeypatch.setattr(factory, "_execution_history_reader", fail_history_reader)
    with pytest.raises(RuntimeError, match="history reader construction failed"):
        await factory.compose_runtime_components(
            _workspace(tmp_path),
            models=ModelRegistry.openai(model="test-model"),
            state=state,
        )
    assert state.ready is False
    assert close_calls == 1


@pytest.mark.asyncio
async def test_compose_build_cleanup_preserves_construction_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._factory as factory

    state = RuntimeState.in_memory()

    class FailingExecutionService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("build secret")

    async def fail_close(_state: RuntimeState) -> None:
        raise RuntimeError("cleanup secret")

    monkeypatch.setattr(factory, "DefaultExecutionService", FailingExecutionService)
    monkeypatch.setattr(RuntimeState, "close", fail_close)
    with pytest.raises(RuntimeError, match="build secret"):
        await factory.compose_runtime_components(
            _workspace(tmp_path),
            models=ModelRegistry.openai(model="test-model"),
            state=state,
        )


@pytest.mark.asyncio
async def test_compose_build_owns_state_after_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._factory as factory

    state = RuntimeState.in_memory()
    close_calls = 0
    original_close = RuntimeState.close

    async def count_close(current: RuntimeState) -> None:
        nonlocal close_calls
        close_calls += 1
        await original_close(current)

    class FailingExecutionService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("build failed")

    monkeypatch.setattr(RuntimeState, "close", count_close)
    monkeypatch.setattr(factory, "DefaultExecutionService", FailingExecutionService)
    with pytest.raises(RuntimeError, match="build failed"):
        await factory.compose_runtime_components(
            _workspace(tmp_path),
            models=ModelRegistry.openai(model="test-model"),
            state=state,
        )
    assert close_calls == 1
    assert state.ready is False


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_name", ("input", "workspace_access"))
async def test_compose_cleanup_continues_after_independent_resource_failure(
    resource_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._factory as factory

    class _CloseCounter:
        def __init__(self, *, fails: bool = False) -> None:
            self.calls = 0
            self._fails = fails

        async def close(self) -> None:
            self.calls += 1
            if self._fails:
                raise RuntimeError("cleanup failure")

    phases: list[str] = []
    monkeypatch.setattr(
        factory,
        "_log_secondary_cleanup",
        lambda phase, _error: phases.append(phase),
    )
    failing = _CloseCounter(fails=True)
    state = _CloseCounter()
    workspace_store = _CloseCounter()
    workspace_backend = _CloseCounter()

    await factory._cleanup_compose_resources(
        selected_state=state,
        initialized=True,
        input_materializer=(
            failing if resource_name == "input" else None
        ),
        workspace_access=(
            failing if resource_name == "workspace_access" else None
        ),
        owned_workspace_assets=(
            workspace_store,
            workspace_backend,
        ),
    )

    assert failing.calls == 1
    assert state.calls == 1
    assert workspace_store.calls == 1
    assert workspace_backend.calls == 1
    assert phases == [f"runtime.compose.{resource_name}"]


@pytest.mark.asyncio
async def test_build_abort_continues_after_input_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._factory as factory

    state = RuntimeState.in_memory()
    state_close_calls = 0
    original_state_close = RuntimeState.close

    class FailingExecutionService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("build failed")

    async def fail_input_close(_input: object) -> None:
        raise RuntimeError("input cleanup failed")

    async def count_state_close(current: RuntimeState) -> None:
        nonlocal state_close_calls
        state_close_calls += 1
        await original_state_close(current)

    monkeypatch.setattr(factory, "DefaultExecutionService", FailingExecutionService)
    monkeypatch.setattr(factory.ExecutionInputMaterializer, "close", fail_input_close)
    monkeypatch.setattr(RuntimeState, "close", count_state_close)

    with pytest.raises(RuntimeError, match="build failed"):
        await factory.compose_runtime_components(
            _workspace(tmp_path),
            models=ModelRegistry.openai(model="test-model"),
            state=state,
        )

    assert state_close_calls == 1
    assert state.ready is False


@pytest.mark.asyncio
async def test_late_build_abort_continues_after_close_action_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._factory as factory

    state = RuntimeState.in_memory()
    state_close_calls = 0
    original_state_close = RuntimeState.close

    async def fail_restore(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("restore failed")

    async def fail_finalizers(_service: object) -> None:
        raise RuntimeError("finalizer cleanup failed")

    async def count_state_close(current: RuntimeState) -> None:
        nonlocal state_close_calls
        state_close_calls += 1
        await original_state_close(current)

    monkeypatch.setattr(factory, "_restore_recovery_bindings", fail_restore)
    monkeypatch.setattr(
        factory.DefaultTaskGraphService,
        "drain_owned_finalizers",
        fail_finalizers,
    )
    monkeypatch.setattr(RuntimeState, "close", count_state_close)

    with pytest.raises(RuntimeError, match="restore failed"):
        await factory.compose_runtime_components(
            _workspace(tmp_path),
            models=ModelRegistry.openai(model="test-model"),
            state=state,
        )

    assert state_close_calls == 1
    assert state.ready is False
