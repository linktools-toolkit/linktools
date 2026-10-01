#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaskGraph cancellation closes remote MCP resources without replaying effects."""

import asyncio
from pathlib import Path

import pytest
from fastmcp import Client
from linktools.ai.core import ExecutionStatus, TaskStatus, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import AgentTaskInput, Runtime, RuntimeStorage
from linktools.ai.task import TaskGraph, TaskNode

from .test_mcp_remote import (
    _group,
    _local_network,  # noqa: F401
    _RemoteModels,
    _serve,
    _spec,
    _wait_closed,
)


@pytest.mark.asyncio
async def test_graph_cancel_closes_remote_client_and_preserves_unknown_effect_on_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_close = Client.close
    closed_clients: list[Client] = []

    async def record_close(client: Client) -> None:
        await original_close(client)
        closed_clients.append(client)

    monkeypatch.setattr(Client, "close", record_close)
    state_path = tmp_path / "runtime"
    async with _serve("sse") as remote:
        remote.block_calls = True
        storage = RuntimeStorage.filesystem(state_path)
        try:
            async with Runtime.open(
                "remote-task-cancellation",
                models=_RemoteModels(),
                storage=storage,
                capabilities=(_group(_spec(remote)),),
            ) as runtime:
                task = runtime.tasks.from_agent(
                    "remote.echo", runtime.agents.get("default")
                )
                graph = TaskGraph(
                    "remote-cancellation",
                    (TaskNode("echo", task=task, input=AgentTaskInput("write once")),),
                )
                run = await runtime.tasks.bind(task).start(
                    graph, idempotency_key="start-remote-cancellation"
                )
                await asyncio.wait_for(remote.call_started.wait(), timeout=10)
                assert remote.active_streams > 0
                assert remote.effects == ["committed"]

                cancelled = await asyncio.wait_for(
                    run.cancel(idempotency_key="cancel-remote-graph"), timeout=10
                )
                assert cancelled.status is TaskStatus.RECOVERY_REQUIRED
                assert closed_clients
                assert all(not client.is_connected() for client in closed_clients)
                remote.release_call.set()
                await _wait_closed(remote)

                state = await run.state(include_content=True)
                assert state.status is TaskStatus.RECOVERY_REQUIRED
                node = state.node_states[0]
                assert node.status is TaskStatus.RECOVERY_REQUIRED
                assert node.execution_id is not None
                execution_id = node.execution_id
                execution = await runtime.executions.get(execution_id)
                effects = await execution.recovery_effects()
                assert len(effects) == 1
                assert effects[0].replay_safe is False
                assert effects[0].error_code == ErrorCode.TOOL_EFFECT_UNKNOWN.value
                operations = await storage.recovery.tools.list_by_execution(
                    execution_id, tenant_id=runtime.default_principal.tenant_id
                )
                assert len(operations) == 1
                assert operations[0].status is ToolOperationStatus.EFFECT_UNKNOWN
                assert remote.effects == ["committed"]
        finally:
            remote.release_call.set()
            await storage.close()

        restored = RuntimeStorage.filesystem(state_path)
        try:
            async with Runtime.open(
                "remote-task-cancellation",
                models=_RemoteModels(),
                storage=restored,
                capabilities=(_group(_spec(remote)),),
            ) as runtime:
                task = runtime.tasks.from_agent(
                    "remote.echo", runtime.agents.get("default")
                )
                run = await runtime.tasks.bind(task).get(graph.graph_id)
                recovered = await run.recover(idempotency_key="recover-remote-graph")
                assert recovered.status is TaskStatus.RECOVERY_REQUIRED
                state = await run.state(include_content=True)
                assert state.node_states[0].execution_id == execution_id
                execution = await runtime.executions.get(execution_id)
                with pytest.raises(AIError) as unresolved:
                    await execution.recover()
                assert unresolved.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
                view = await runtime.executions.inspect(
                    execution_id, principal=runtime.default_principal
                )
                assert view.status is ExecutionStatus.RECOVERY_REQUIRED
                assert await execution.recovery_effects() == effects
                assert remote.effects == ["committed"]
                assert remote.methods.count("tools/call") == 1
        finally:
            await restored.close()
        await _wait_closed(remote)
