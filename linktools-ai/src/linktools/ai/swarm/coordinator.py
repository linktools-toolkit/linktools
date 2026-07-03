#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmCoordinator: run N agent instances against one shared TaskQueue
until it's drained. Mirrors AgentKernel.start_background's construction
pattern (core/runtime.py) for spinning up a SubAgent programmatically."""

import asyncio
from typing import TYPE_CHECKING, Callable

from ..subagent.registry import SubagentSpec
from .protocols import Task, TaskQueue

if TYPE_CHECKING:
    from ..core.model_runtime import RuntimeModelConfig
    from ..core.runtime import AgentKernel
    from ..session.types import Session


class SwarmCoordinator:
    def __init__(
        self,
        kernel: "AgentKernel",
        task_queue: TaskQueue,
        model_config_resolver: "Callable[[str], RuntimeModelConfig]",
    ) -> None:
        self.kernel = kernel
        self.task_queue = task_queue
        self.model_config_resolver = model_config_resolver

    async def run(self, spec: SubagentSpec, session: "Session", *, agent_count: int) -> "list[Task]":
        # Deferred import: agent.py imports AgentKernel from core/runtime.py
        # at module level, so importing SubAgent here at module level would be
        # circular (same hazard AgentKernel.start_background already works around).
        from ..agent import SubAgent

        async def _worker(agent_id: str) -> None:
            while True:
                task = await self.task_queue.claim(agent_id)
                if task is None:
                    return
                try:
                    child_context = self.kernel.build_context(
                        spec, session, builtin_tool_names=SubAgent._BUILTIN_TOOL_NAMES,
                    )
                    agent = SubAgent(spec, session, execution_context=child_context, model_config_resolver=self.model_config_resolver)
                    result = await agent.generate(task.payload, call_id=task.task_id)
                    await self.task_queue.complete(task.task_id, result)
                except Exception as exc:
                    await self.task_queue.fail(task.task_id, str(exc))

        await asyncio.gather(*[_worker(f"swarm-agent-{i}") for i in range(agent_count)])
        return await self.task_queue.list()
