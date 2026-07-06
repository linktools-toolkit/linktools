#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentCompiler: resolves an AgentSpec's model via ModelRouter and builds the
underlying pydantic-ai Agent. Entirely stateless -- never touches Session, Run,
or the filesystem. Per review-doc §17, the compiler accepts no working-directory
or ExecutionBackend parameter and never constructs ``LocalExecutionBackend``:
builtin file/terminal tools are constructed at EXECUTION TIME from
``AgentDependencies.execution`` and passed to ``agent.iter(prompt, toolsets=)``.
The compiled Agent carries model + capabilities (policy + middleware) only."""

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


class AgentCompiler:
    def __init__(
        self,
        *,
        model_router: ModelRouter,
        tool_executor: "ToolExecutor | None" = None,
        middleware_pipeline: "MiddlewarePipeline | None" = None,
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

    async def compile(self, spec: AgentSpec) -> CompiledAgent:
        bundle = await self._model_router.resolve(spec.model)
        capability = build_policy_capability(self._tool_executor)
        capabilities = [capability]
        if self._middleware_pipeline is not None:
            middleware_capability = build_middleware_capability(self._middleware_pipeline)
            capabilities.append(middleware_capability)
        else:
            middleware_capability = None
        pydantic_agent = PydanticAgent(
            bundle.model,
            output_type=spec.output_schema or dict,
            capabilities=capabilities,
            deps_type=AgentDependencies,
        )
        return CompiledAgent(
            spec=spec,
            pydantic_agent=pydantic_agent,
            model_bundle=bundle,
            policy_capability=capability,
            middleware_capability=middleware_capability,
        )
