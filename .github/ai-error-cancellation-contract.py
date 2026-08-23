"""Cancellation and capability error-boundary regressions."""

import asyncio
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from linktools.ai.agent import SubagentDelegate
from linktools.ai.asset import AssetRef
from linktools.ai.asset._sql import SqlAssetBackend
from linktools.ai.capability import CapabilityMaterializationContext
from linktools.ai.capability._mcp import bind_mcp_capability
from linktools.ai.core import ExecutionStatus, JsonValue, ResourceKind, ResourceRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._evaluation import DefaultEvaluationService
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime._planner import RuntimeTaskNodeRunner
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.runtime._subagent import SubagentDispatcher
from linktools.ai.spec import MCPServerSpec
from linktools.ai.task._service_impl import DefaultTaskService
from linktools.ai.workspace import trusted_workspace_principal


def test_subagent_delegate_contract_requires_mapping_result() -> None:
    assert get_type_hints(SubagentDelegate.__call__)["return"] == dict[str, JsonValue]


@pytest.mark.asyncio
async def test_mcp_runtime_shape_violation_is_capability_error(tmp_path) -> None:
    class MalformedRuntime:
        @property
        def fingerprint(self) -> str:
            return "a" * 64

        async def toolsets(self, servers, **kwargs):
            del servers, kwargs
            return ()

    binding = bind_mcp_capability(
        (AssetRef("mcp", "server"),),
        (MCPServerSpec("server", 1, "echo"),),
        MalformedRuntime(),
    )
    context = CapabilityMaterializationContext(
        trusted_workspace_principal("tenant"),
        ResourceRef(ResourceKind.EXECUTION, "execution", "tenant"),
        tmp_path,
    )

    with pytest.raises(AIError) as error:
        await binding.materialize(context)

    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_asset_sql_apply_preserves_cancellation(monkeypatch) -> None:
    async def cancelled(self, changes, expected_revision):
        del self, changes, expected_revision
        raise asyncio.CancelledError

    monkeypatch.setattr(SqlAssetBackend, "_apply_once_transaction", cancelled)
    backend = object.__new__(SqlAssetBackend)

    with pytest.raises(asyncio.CancelledError):
        await backend._apply_once((), None)


@pytest.mark.asyncio
async def test_execution_handoff_cleanup_restores_state_before_cancellation() -> None:
    service = object.__new__(DefaultExecutionService)
    service._handoff_condition = asyncio.Condition()

    async def cancelled(execution_id: str, *, tenant_id: str) -> None:
        del execution_id, tenant_id
        raise asyncio.CancelledError

    service._release_terminal = cancelled
    state = SimpleNamespace(release_in_progress=True, release_requested=False)

    with pytest.raises(asyncio.CancelledError):
        await service._run_handoff_cleanup("execution", "tenant", state)

    assert state.release_in_progress is False
    assert state.release_requested is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service_type", "consumer_name", "resource_id"),
    (
        (DefaultEvaluationService, "_evaluation_consumer", "evaluation"),
        (DefaultSessionService, "_session_consumer", "session"),
        (DefaultTaskService, "_graph_consumer", "graph"),
    ),
)
async def test_transient_handoff_cleanup_preserves_cancellation(
    service_type,
    consumer_name: str,
    resource_id: str,
) -> None:
    service = object.__new__(service_type)
    service._handoff_condition = asyncio.Condition()
    service._handoff_states = {}

    async def cancelled(*args, **kwargs) -> None:
        del args, kwargs
        raise asyncio.CancelledError

    service._release_terminal = cancelled
    key = ("tenant", resource_id)

    with pytest.raises(asyncio.CancelledError):
        async with getattr(service, consumer_name)(resource_id, "tenant"):
            service._handoff_states[key].release_requested = True

    state = service._handoff_states[key]
    assert state.release_in_progress is False
    assert state.release_requested is True


@pytest.mark.asyncio
async def test_local_terminal_admission_cleanup_preserves_cancellation() -> None:
    class Sessions:
        async def release_execution(self, *args, **kwargs) -> None:
            del args, kwargs
            raise asyncio.CancelledError

    backend = object.__new__(LocalExecutionBackend)
    backend._conversation = SimpleNamespace(sessions=Sessions())
    execution = SimpleNamespace(
        session_id="session",
        tenant_id="tenant",
        execution_id="execution",
    )

    with pytest.raises(asyncio.CancelledError):
        await backend._release_session_admission_best_effort(execution)


@pytest.mark.asyncio
async def test_task_runner_preserves_cancellation_when_child_cleanup_fails() -> None:
    class Execution:
        def __init__(self) -> None:
            self.wait_started = asyncio.Event()

        async def run(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(execution_id="execution")

        async def wait(self, *args, **kwargs):
            del args, kwargs
            self.wait_started.set()
            await asyncio.Event().wait()

        async def cancel(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("cleanup failed")

    execution = Execution()
    runner = object.__new__(RuntimeTaskNodeRunner)
    runner._execution = execution

    async def prepare(*args, **kwargs):
        del args, kwargs
        return "binding", object()

    runner.prepare = prepare
    task = asyncio.create_task(
        runner.run(
            SimpleNamespace(node_id="node"),
            graph_id="graph",
            principal=trusted_workspace_principal("tenant"),
            dependency_results={},
        )
    )
    await execution.wait_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as error:
        await task

    assert isinstance(error.value.__cause__, AIError)
    assert error.value.__cause__.code is ErrorCode.STORAGE_RECOVERY_REQUIRED


@pytest.mark.asyncio
async def test_subagent_child_cleanup_failure_does_not_replace_cancellation() -> None:
    class Execution:
        def __init__(self) -> None:
            self.cancel_started = asyncio.Event()
            self.finish_cancel = asyncio.Event()

        async def inspect(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(status=ExecutionStatus.RUNNING)

        async def cancel(self, *args, **kwargs):
            del args, kwargs
            self.cancel_started.set()
            await self.finish_cancel.wait()
            raise RuntimeError("cleanup failed")

    execution = Execution()
    dispatcher = object.__new__(SubagentDispatcher)
    dispatcher._execution = execution
    task = asyncio.create_task(
        dispatcher.cancel_child(
            "execution",
            parent_execution_id="parent",
            principal=trusted_workspace_principal("tenant"),
        )
    )
    await execution.cancel_started.wait()
    task.cancel()
    execution.finish_cancel.set()

    with pytest.raises(asyncio.CancelledError) as error:
        await task

    assert isinstance(error.value.__cause__, RuntimeError)
