#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentCompiler: resolves an AgentSpec's ModelPolicy to a ResolvedModel and
builds the underlying pydantic-ai Agent on that real model. Entirely stateless
-- never touches Session, Run, or the filesystem. The compiler accepts no
working-directory or Sandbox parameter and never constructs ``LocalSandbox``:
builtin file/terminal tools are constructed at EXECUTION TIME from
``AgentDependencies.sandbox`` and passed to ``agent.iter(prompt, toolsets=)``.
The compiled Agent carries model + SDK hooks (policy + middleware) + the
spec's static instructions (``PromptSpec.instructions``) only.

The compiler never bakes in a default command denylist. The default
SecurityBaseline (including its CommandRule) is resolved exactly once, by
``build_runtime`` -- the compiler only ever consumes the ``tool_executor`` it
is given. ``tool_executor`` is OPTIONAL: when omitted, the compiled Agent
carries no ``PolicyCapability`` (no command governance). The engine still
rejects a tool-less run that actually needs tools, so a compiler without an
executor is legal for tool-free agents and fails loudly at execution time
rather than silently governing nothing."""


from .middleware.capability import build_middleware_capability
from .middleware.pipeline import MiddlewarePipeline
from .tool.pydantic_ai import build_policy_capability
from .dependencies import AgentDependencies
from .models import CompiledAgent
from .codec import OutputTypeRegistry
from pydantic_ai import Agent as PydanticAgent

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model.resolver import ModelResolver, ResolvedModel
    from .spec import AgentSpec

class AgentCompiler:
    def __init__(
        self,
        *,
        model_resolver: "ModelResolver",
        tool_executor: "object | None" = None,
        middleware_pipeline: "MiddlewarePipeline | None" = None,
        output_types: "OutputTypeRegistry | None" = None,
    ) -> None:
        self._model_resolver = model_resolver
        self._tool_executor = tool_executor
        self._middleware_pipeline = middleware_pipeline
        self._output_types = output_types

    async def compile(self, spec: "AgentSpec") -> CompiledAgent:
        resolved: "ResolvedModel" = self._model_resolver.resolve(spec.model)
        capability = build_policy_capability(self._tool_executor) if self._tool_executor is not None else None
        capabilities = [capability] if capability is not None else []
        if self._middleware_pipeline is not None:
            middleware_capability = build_middleware_capability(
                self._middleware_pipeline
            )
            capabilities.append(middleware_capability)
        else:
            middleware_capability = None
        pydantic_agent = PydanticAgent(
            resolved.model,
            output_type=spec.output_schema or str,
            capabilities=capabilities,
            deps_type=AgentDependencies,
            instructions=spec.instructions.instructions,
        )
        return CompiledAgent(
            spec=spec,
            pydantic_agent=pydantic_agent,
            model_bundle=resolved,
            policy_capability=capability,
            middleware_capability=middleware_capability,
        )
