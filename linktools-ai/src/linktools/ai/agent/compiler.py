#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentCompiler: resolves an AgentSpec's model via ModelRouter and builds the
underlying pydantic-ai Agent. Entirely stateless -- never touches Session or Run.
A `workdir` (when provided) is the only filesystem surface this compiler
touches: it scopes the builtin file/terminal toolset so compiled agents can
read/write/list/patch files and run bash against that directory."""

from pathlib import Path
from typing import TYPE_CHECKING

from ..middleware.capability import build_middleware_capability
from ..middleware.pipeline import MiddlewarePipeline
from ..model.router import ModelRouter
from ..policy.command import CommandRule, DEFAULT_DENIED_COMMAND_PATTERNS
from ..policy.engine import PolicyEngine
from ..tool.capability import build_policy_capability
from ..tool.executor import ToolExecutor
from .dependencies import AgentDependencies
from .models import CompiledAgent
from .spec import AgentSpec

from pydantic_ai import Agent as PydanticAgent

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset


class AgentCompiler:
    def __init__(
        self,
        *,
        model_router: ModelRouter,
        tool_executor: "ToolExecutor | None" = None,
        middleware_pipeline: "MiddlewarePipeline | None" = None,
        workdir: "Path | None" = None,
        pause_on_approval: bool = False,
    ) -> None:
        self._model_router = model_router
        # When the caller does not supply an explicit ``tool_executor``, build
        # the default CommandRule-only executor. ``pause_on_approval`` threads
        # through so a directly-constructed compiler can opt into the pause
        # path -- note, however, that pausing without an ``approval_store``
        # wired falls through to the legacy ToolApprovalRequiredError raise
        # (Task 9 wires the store at the Runtime.build level where Storage is
        # available). When ``tool_executor`` is explicit the flag is purely
        # informational: the supplied executor already carries its own setting.
        self._pause_on_approval = pause_on_approval
        self._tool_executor = tool_executor or ToolExecutor(
            policy=PolicyEngine(rules=(CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS),)),
            pause_on_approval=pause_on_approval,
        )
        self._middleware_pipeline = middleware_pipeline
        # When set, the compiled pydantic-ai Agent carries a builtin
        # FunctionToolset (list_dir/read_file/write_file/batch_files/apply_patch
        # + bash) backed by a LocalExecutionBackend rooted at this directory.
        # ``None`` (default) keeps the compiler stateless and registers no
        # builtin tools -- existing compiler tests rely on that contract.
        self._workdir = workdir

    async def compile(self, spec: AgentSpec) -> CompiledAgent:
        bundle = await self._model_router.resolve(spec.model)
        capability = build_policy_capability(self._tool_executor)
        capabilities = [capability]
        if self._middleware_pipeline is not None:
            middleware_capability = build_middleware_capability(self._middleware_pipeline)
            capabilities.append(middleware_capability)
        else:
            middleware_capability = None
        toolsets: "list[AbstractToolset]" = []
        if self._workdir is not None:
            from ..execution.local import LocalExecutionBackend
            from ..execution.toolset import BuiltinToolContext, build_builtin_toolset

            backend = LocalExecutionBackend(runtime_dir=self._workdir)
            ctx = BuiltinToolContext(backend=backend, enabled_tools={"file", "terminal"})
            toolsets.append(build_builtin_toolset(ctx))
        pydantic_agent = PydanticAgent(
            bundle.model,
            output_type=spec.output_schema or dict,
            capabilities=capabilities,
            toolsets=toolsets or None,
            deps_type=AgentDependencies,
        )
        return CompiledAgent(
            spec=spec,
            pydantic_agent=pydantic_agent,
            model_bundle=bundle,
            policy_capability=capability,
            middleware_capability=middleware_capability,
        )
