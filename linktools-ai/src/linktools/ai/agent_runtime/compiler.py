#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentCompiler: resolves an AgentSpec's model via ModelRouter and builds the
underlying pydantic-ai Agent. Entirely stateless -- never touches Session, Run,
or a working directory."""

from ..model.router import ModelRouter
from ..policy.command import CommandRule, DEFAULT_DENIED_COMMAND_PATTERNS
from ..policy.engine import PolicyEngine
from ..tool.capability import build_policy_capability
from ..tool.executor import ToolExecutor
from .models import CompiledAgent
from .spec import AgentSpec

from pydantic_ai import Agent as PydanticAgent


class AgentCompiler:
    def __init__(self, *, model_router: ModelRouter, tool_executor: "ToolExecutor | None" = None) -> None:
        self._model_router = model_router
        self._tool_executor = tool_executor or ToolExecutor(
            policy=PolicyEngine(rules=(CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS),))
        )

    async def compile(self, spec: AgentSpec) -> CompiledAgent:
        bundle = await self._model_router.resolve(spec.model)
        capability = build_policy_capability(self._tool_executor)
        pydantic_agent = PydanticAgent(
            bundle.model,
            output_type=spec.output_schema or dict,
            capabilities=[capability],
        )
        return CompiledAgent(spec=spec, pydantic_agent=pydantic_agent, model_bundle=bundle, policy_capability=capability)
