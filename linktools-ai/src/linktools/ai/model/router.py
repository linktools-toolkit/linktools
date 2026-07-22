#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model resolution split into two concerns:

- :class:`ModelResolver` resolves a SINGLE model_type to its configured
  :class:`ModelBundle` -- pure config resolution, one attempt, no retry and no
  fallback. It raises :class:`ModelClientUnavailable` if that one type is not
  registered.

- :class:`ModelGateway` is the resilience wrapper the run path uses: given a
  :class:`ModelPolicy` it walks primary then each fallback, retrying each
  candidate up to ``max_retries`` times (catching ``ModelClientUnavailable``)
  before moving on. Total attempts per resolve =
  ``(1 + max_retries) * len(attempted)``; ``max_retries=0`` yields exactly one
  attempt per candidate. This is where fallback + retry live; the resolver
  never decides policy.

Separating them keeps the registry-lookup concern (ModelResolver) testable in
isolation from the retry/fallback policy (ModelGateway)."""

from ..model.registry import (
    ModelBundle,
    ModelClientUnavailable,
    ModelRegistry,
    model_registry,
)
from ..errors import ModelRoutingError
from .policy import ModelPolicy


class ModelResolver:
    """Resolve a single model_type to its configured ModelBundle. One attempt,
    no retry, no fallback: raising :class:`ModelClientUnavailable` is the signal
    a caller (the :class:`ModelGateway`) retries or falls back on."""

    def __init__(self, *, registry: ModelRegistry = model_registry) -> None:
        self._registry = registry

    async def resolve(self, model_type: str) -> ModelBundle:
        return self._registry.get(model_type)


class ModelGateway:
    """The model-policy resilience layer: resolve a :class:`ModelPolicy` by
    walking primary then fallbacks, retrying each candidate up to
    ``max_retries`` times via the wrapped :class:`ModelResolver` before giving
    up. Raises :class:`ModelRoutingError` only when every candidate is
    exhausted."""

    def __init__(self, resolver: ModelResolver) -> None:
        self._resolver = resolver

    async def resolve(self, policy: ModelPolicy) -> ModelBundle:
        attempted = [policy.primary, *policy.fallbacks]
        last_error: "Exception | None" = None
        for model_type in attempted:
            for _attempt in range(policy.max_retries + 1):
                try:
                    return await self._resolver.resolve(model_type)
                except ModelClientUnavailable as exc:
                    last_error = exc
        raise ModelRoutingError(
            f"no model available among {attempted!r}"
        ) from last_error


__all__: "list[str]" = ["ModelResolver", "ModelGateway", "ModelRoutingError"]
