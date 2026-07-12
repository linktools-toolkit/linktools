#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentCompiler: resolves an AgentSpec's model via ModelRouter and builds the
underlying pydantic-ai Agent. Entirely stateless -- never touches Session, Run,
or the filesystem. The compiler accepts no working-directory
or ExecutionBackend parameter and never constructs ``LocalExecutionBackend``:
builtin file/terminal tools are constructed at EXECUTION TIME from
``AgentDependencies.execution`` and passed to ``agent.iter(prompt, toolsets=)``.
The compiled Agent carries model + capabilities (policy + middleware) only.

The compiler never bakes in a default command denylist. The default
SecurityBaseline (including its CommandRule) is resolved exactly once, by
``Runtime.build`` -- the compiler only ever consumes whatever ``tool_executor``
it is given. A directly-constructed compiler with no explicit ``tool_executor``
gets an inert, rule-less PolicyEngine (ALLOW everything) rather than a
silently-reinstated default -- callers that want the baseline denylist go
through ``Runtime.build``, the single source of truth for it."""

from ..middleware.capability import build_middleware_capability
from ..middleware.pipeline import MiddlewarePipeline
from ..model.router import ModelRouter
from ..policy.engine import PolicyEngine
from ..tool.pydantic import build_policy_capability
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
        # ``pause_on_approval`` threads through so a directly-constructed
        # compiler can opt into the pause path when it also builds its own
        # (rule-less) executor below -- note, however, that pausing without an
        # ``approval_store`` wired falls through to the prior
        # ToolApprovalRequiredError raise. When ``tool_executor`` is explicit
        # the flag is purely informational: the supplied executor already
        # carries its own setting.
        self._pause_on_approval = pause_on_approval
        self._tool_executor = tool_executor or ToolExecutor(
            policy=PolicyEngine(rules=()),
            pause_on_approval=pause_on_approval,
        )
        self._middleware_pipeline = middleware_pipeline

    async def compile(self, spec: AgentSpec) -> CompiledAgent:
        bundle = await self._model_router.resolve(spec.model)
        capability = build_policy_capability(self._tool_executor)
        capabilities = [capability]
        if self._middleware_pipeline is not None:
            middleware_capability = build_middleware_capability(
                self._middleware_pipeline
            )
            capabilities.append(middleware_capability)
        else:
            middleware_capability = None
        pydantic_agent = PydanticAgent(
            bundle.model,
            output_type=spec.output_schema or str,
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
