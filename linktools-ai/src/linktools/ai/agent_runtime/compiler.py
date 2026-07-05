#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentCompiler: resolves an AgentSpec's model via ModelRouter and builds the
underlying pydantic-ai Agent. Entirely stateless -- never touches Session, Run,
or a working directory."""

from ..middleware.capability import build_middleware_capability
from ..middleware.pipeline import MiddlewarePipeline
from ..model.router import ModelRouter
from ..policy.command import CommandRule, DEFAULT_DENIED_COMMAND_PATTERNS
from ..policy.engine import PolicyEngine
from ..tool.capability import build_policy_capability
from ..tool.executor import ToolExecutor
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
    ) -> None:
        self._model_router = model_router
        self._tool_executor = tool_executor or ToolExecutor(
            policy=PolicyEngine(rules=(CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS),))
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
        )
        return CompiledAgent(
            spec=spec,
            pydantic_agent=pydantic_agent,
            model_bundle=bundle,
            policy_capability=capability,
            middleware_capability=middleware_capability,
        )
