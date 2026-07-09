#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ModelRouter: resolves a ModelPolicy to a ModelBundle, trying primary then each
fallback in order. Makes ModelPolicy.fallbacks load-bearing -- the pre-vNext
BaseAgent.fallback_models field was accepted but never read anywhere.

GAP-08 (spec 31): ModelPolicy.max_retries is now enforced -- each model_type is
retried up to ``max_retries`` times (catching ModelClientUnavailable) before the
router moves on to the next fallback. Total attempts per resolve =
``(1 + max_retries) * len(attempted)``. The default max_retries=0 reproduces the
pre-GAP-08 behavior (exactly one attempt per model_type)."""

from ..model.registry import ModelBundle, ModelClientUnavailable, ModelRegistry, model_registry
from ..errors import ModelRoutingError
from .policy import ModelPolicy


class ModelRouter:
    def __init__(self, *, registry: ModelRegistry = model_registry) -> None:
        self._registry = registry

    async def resolve(self, policy: ModelPolicy) -> ModelBundle:
        attempted = [policy.primary, *policy.fallbacks]
        last_error: "Exception | None" = None
        # GAP-08: retry each model_type up to max_retries times before falling
        # back. max_retries=0 -> one attempt per model_type (legacy behavior).
        for model_type in attempted:
            for _attempt in range(policy.max_retries + 1):
                try:
                    return self._registry.get(model_type)
                except ModelClientUnavailable as exc:
                    last_error = exc
        raise ModelRoutingError(
            f"no model available among {attempted!r}"
        ) from last_error
