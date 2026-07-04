#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ForkCoordinator: branch execution + result collection, no auto-merge.

Session.copy() alone is NOT sufficient for fork isolation -- it gives each
branch its own root/transcript, but concurrent branches must not collide
when writing files or running shell commands, so this coordinator
additionally forks the ExecutionBackend (execution/local.py's
LocalExecutionBackend.fork(), which copies `workdir` into an isolated
directory per branch) and constructs each branch's SubAgent with
`workdir=branch_workdir` directly. `workdir` belongs to the agent, not the
session (see agent.py's RuntimeAgent), so there's no Session/RunContext
mutation needed here anymore -- just pass the forked directory straight to
the constructor.

Branch working directories are created as siblings of `workdir`, under
`workdir.parent / f".{workdir.name}.forks"`, rather than nested underneath
`workdir` itself. `LocalExecutionBackend.fork()` copies `workdir` (the
source) into `branch_workdir` (the destination) via `shutil.copytree`; if
the destination were nested inside the source, each copy would include the
partially-written destination and recurse into itself, growing without
bound. Keeping the forks directory as a sibling guarantees the destination
can never be a descendant of the source."""

import asyncio
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..execution.local import LocalExecutionBackend
from ..subagent.registry import SubagentSpec

if TYPE_CHECKING:
    from ..core.runtime import AgentKernel
    from ..session.types import Session


class ForkCoordinator:
    def __init__(self, kernel: "AgentKernel") -> None:
        self.kernel = kernel

    async def run(
        self, spec: SubagentSpec, session: "Session", inputs: Any, *, branch_count: int, workdir: Path,
    ) -> "list[dict[str, Any]]":
        from ..agent import SubAgent

        async def _run_branch(index: int) -> "dict[str, Any]":
            branch_id = f"branch-{index}-{uuid.uuid4().hex[:8]}"
            forks_root = workdir.parent / f".{workdir.name}.forks"
            branch_workdir = forks_root / session.session_id / branch_id

            branch_session = session.copy(child_session_id=f"{session.session_id}-{branch_id}")

            try:
                parent_backend = LocalExecutionBackend(runtime_dir=workdir, base_dirs=[])
                await parent_backend.fork(branch_workdir)

                child_context = self.kernel.build_context(
                    spec, branch_session, builtin_tool_names=SubAgent._BUILTIN_TOOL_NAMES,
                )
                agent = SubAgent(
                    spec, branch_session, execution_context=child_context,
                    workdir=branch_workdir,
                )
                result = await agent.generate(inputs, call_id=branch_id)
                return {"branch_id": branch_id, "status": "done", "result": result, "error": None}
            except Exception as exc:
                return {"branch_id": branch_id, "status": "failed", "result": None, "error": str(exc)}

        return list(await asyncio.gather(*[_run_branch(i) for i in range(branch_count)]))
