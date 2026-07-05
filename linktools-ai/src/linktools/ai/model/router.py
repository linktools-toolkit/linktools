#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ModelRouter: resolves a ModelPolicy to a ModelBundle, trying primary then each
fallback in order. Makes ModelPolicy.fallbacks load-bearing -- the pre-vNext
BaseAgent.fallback_models field was accepted but never read anywhere."""

from ..core.model_runtime import ModelBundle, ModelClientUnavailable, ModelRegistry, model_registry
from ..errors import ModelRoutingError
from .policy import ModelPolicy


class ModelRouter:
    def __init__(self, *, registry: ModelRegistry = model_registry) -> None:
        self._registry = registry

    async def resolve(self, policy: ModelPolicy) -> ModelBundle:
        attempted = [policy.primary, *policy.fallbacks]
        last_error: "Exception | None" = None
        for model_type in attempted:
            try:
                return self._registry.get(model_type)
            except ModelClientUnavailable as exc:
                last_error = exc
                continue
        raise ModelRoutingError(
            f"no model available among {attempted!r}"
        ) from last_error
