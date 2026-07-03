#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ForkCoordinator: branch execution + result collection, no auto-merge.

Session.copy() alone is NOT sufficient for fork isolation -- it gives each
branch its own session_dir/transcript but keeps the SAME runtime_dir as the
parent (verified directly against FileSession.copy in session/types.py).
Concurrent branches must not collide when writing files or running shell
commands, so this coordinator additionally forks the ExecutionBackend
(execution/local.py's LocalExecutionBackend.fork(), which copies
runtime_dir into an isolated directory per branch), then overrides the
branch Session's `run.runtime_dir` to point at that forked directory --
`_build_model_agent` (core/agent.py) always constructs its own
LocalExecutionBackend internally from `self.session.run.runtime_dir` (see
core/agent.py around the `_build_model_agent` method), it never accepts an
externally-constructed backend. So the only way to make a SubAgent's actual
tool calls land in the forked directory is to point `session.run.runtime_dir`
there before constructing the SubAgent -- forking the backend alone, without
also repointing the Session, would fork a directory nothing ever reads from.
Verified directly: `dataclasses.replace(branch.run, runtime_dir=...)` then
`dataclasses.replace(branch, run=new_run)` works since both RunContext and
FileSession are plain (non-frozen-check-required) dataclasses with a `run`
field -- confirmed via `dataclasses.fields(FileSession)`."""

import asyncio
import dataclasses
import uuid
from typing import TYPE_CHECKING, Any, Callable

from ..execution.local import LocalExecutionBackend
from ..subagent.registry import SubagentSpec

if TYPE_CHECKING:
    from ..core.model_runtime import RuntimeModelConfig
    from ..core.runtime import AgentKernel
    from ..session.types import Session


class ForkCoordinator:
    def __init__(
        self,
        kernel: "AgentKernel",
        model_config_resolver: "Callable[[str], RuntimeModelConfig]",
    ) -> None:
        self.kernel = kernel
        self.model_config_resolver = model_config_resolver

    async def run(
        self, spec: SubagentSpec, session: "Session", inputs: Any, *, branch_count: int,
    ) -> "list[dict[str, Any]]":
        from ..agent import SubAgent

        async def _run_branch(index: int) -> "dict[str, Any]":
            branch_id = f"branch-{index}-{uuid.uuid4().hex[:8]}"
            branch_runtime_dir = session.workspace_root / "forks" / session.session_id / branch_id

            parent_backend = LocalExecutionBackend(runtime_dir=session.runtime_dir, base_dirs=[])
            await parent_backend.fork(branch_runtime_dir)

            branch_session = session.copy(child_session_id=f"{session.session_id}-{branch_id}")
            isolated_run = dataclasses.replace(branch_session.run, runtime_dir=branch_runtime_dir)
            branch_session = dataclasses.replace(branch_session, run=isolated_run)

            try:
                child_context = self.kernel.build_context(
                    spec, branch_session, builtin_tool_names=SubAgent._BUILTIN_TOOL_NAMES,
                )
                agent = SubAgent(spec, branch_session, execution_context=child_context, model_config_resolver=self.model_config_resolver)
                result = await agent.generate(inputs, call_id=branch_id)
                return {"branch_id": branch_id, "status": "done", "result": result, "error": None}
            except Exception as exc:
                return {"branch_id": branch_id, "status": "failed", "result": None, "error": str(exc)}

        return list(await asyncio.gather(*[_run_branch(i) for i in range(branch_count)]))
