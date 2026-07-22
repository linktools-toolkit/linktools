#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentCompiler: resolves an AgentSpec's model via the ModelGateway and builds
the underlying pydantic-ai Agent. Entirely stateless -- never touches Session,
Run, or the filesystem. The compiler accepts no working-directory
or Sandbox parameter and never constructs ``LocalSandbox``:
builtin file/terminal tools are constructed at EXECUTION TIME from
``AgentDependencies.execution`` and passed to ``agent.iter(prompt, toolsets=)``.
The compiled Agent carries model + capabilities (policy + middleware) + the
spec's static instructions (``PromptSpec.instructions``) only.

The compiler never bakes in a default command denylist. The default
SecurityBaseline (including its CommandRule) is resolved exactly once, by
``Runtime.build`` -- the compiler only ever consumes the ``tool_executor`` it
is given, which is REQUIRED: there is no rule-less ALLOW-all fallback, so a
directly-constructed compiler without an explicit executor fails loudly rather
than silently governing nothing."""

from ..errors import RuntimeInitializationError
from ..middleware.capability import build_middleware_capability
from ..middleware.pipeline import MiddlewarePipeline
from ..model.router import ModelGateway
from ..tool.pydantic import build_policy_capability
from ..tool.executor import GovernedToolInvoker
from .dependencies import AgentDependencies
from .models import CompiledAgent
from .spec import AgentSpec

from pydantic_ai import Agent as PydanticAgent


class AgentCompiler:
    def __init__(
        self,
        *,
        model_router: ModelGateway,
        tool_executor: GovernedToolInvoker,
        middleware_pipeline: "MiddlewarePipeline | None" = None,
    ) -> None:
        if tool_executor is None:
            raise RuntimeInitializationError(
                "AgentCompiler requires a GovernedToolInvoker; Runtime.build is the "
                "single source of the baseline-governed executor"
            )
        self._model_router = model_router
        self._tool_executor = tool_executor
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
            instructions=spec.instructions.instructions,
        )
        return CompiledAgent(
            spec=spec,
            pydantic_agent=pydantic_agent,
            model_bundle=bundle,
            policy_capability=capability,
            middleware_capability=middleware_capability,
        )
