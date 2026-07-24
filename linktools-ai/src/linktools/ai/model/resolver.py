#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve a :class:`ModelPolicy` to a :class:`ResolvedModel` carrying the real
pydantic-ai Model to inject into the Agent.

Fallback lives at the REQUEST layer, not the registry-lookup layer. The resolver
walks primary then fallbacks ONCE -- a registry lookup is an in-memory read, not
a network call, so retrying it was meaningless. Unregistered candidates are
skipped (diagnostic, not fatal). With
two or more registered candidates, their models are wrapped in pydantic-ai's
:class:`~pydantic_ai.models.fallback.FallbackModel`, which tries each model at
request time and advances on :class:`~pydantic_ai.exceptions.ModelHTTPError`; a
single candidate is used directly.

``request_retries`` configures the provider HTTP client's own retry of transient
HTTP failures (wired into ``AsyncOpenAI`` as ``max_retries`` when a config-backed
OpenAI model is built, ALWAYS explicitly including 0). It is NOT a registry-lookup
retry. ``None`` is the signal that a prebuilt model (registered directly via
``model=``) manages its own retry behavior and is reused as-is; an int
``request_retries`` on a prebuilt model is rejected (the framework cannot configure
a client it does not own). The resolved revision is a stable hash of the ordered
candidates' non-secret identity plus ``request_retries``: reordering the chain,
swapping an endpoint field, or changing the retry count are real revision changes;
rotating an api_key is not."""

import hashlib
from dataclasses import dataclass

from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from ..errors import ModelRoutingError, ModelRetryConfigurationError
from ..json import canonical_json
from .policy import ModelPolicy
from .registry import ModelBundle, ModelClientUnavailable, ModelRegistry, model_registry


@dataclass(frozen=True)
class ResolvedModel:
    """The real model + its stable revision + per-call limits, ready to inject
    into a pydantic-ai Agent."""

    model: Model
    revision: str
    usage_limits: UsageLimits


class ModelResolver:
    """Resolve a :class:`ModelPolicy` to a :class:`ResolvedModel` by walking the
    candidate chain once and, for multiple registered candidates, wrapping them
    in a pydantic-ai ``FallbackModel`` (request-layer fallback)."""

    def __init__(self, *, registry: ModelRegistry = model_registry) -> None:
        self._registry = registry

    def resolve(self, policy: ModelPolicy) -> ResolvedModel:
        bundles: "list[ModelBundle]" = []
        for model_type in (policy.primary, *policy.fallbacks):
            try:
                bundles.append(self._registry.get(model_type))
            except ModelClientUnavailable:
                # Unregistered candidate: skip (diagnostic, not fatal). A later
                # candidate may still satisfy the policy.
                continue
        if not bundles:
            raise ModelRoutingError(
                f"no registered model among primary={policy.primary!r} "
                f"fallbacks={policy.fallbacks!r}"
            )
        models = [_candidate_model(b, policy.request_retries) for b in bundles]
        revision = _resolved_revision(bundles, policy.request_retries)
        usage_limits = bundles[0].usage_limits
        if len(models) == 1:
            return ResolvedModel(models[0], revision, usage_limits)
        from pydantic_ai.models.fallback import FallbackModel

        return ResolvedModel(FallbackModel(*models), revision, usage_limits)


def _candidate_model(bundle: ModelBundle, request_retries: "int | None") -> Model:
    """Pick the model for one candidate, applying the policy's retry semantics.

    Config-backed OpenAI model: the framework ALWAYS owns ``max_retries``. A
    registration builds with ``max_retries=0``; when the policy asks for a
    positive count the model is rebuilt so the provider client carries that count
    (0 is already explicit at registration, so the registered model is reused).
    ``request_retries=None`` on a config-backed model is treated as 0 (the
    config-file default) -- None is the prebuilt signal, but a config-backed model
    has a framework-managed client either way.

    Prebuilt model: it owns its own HTTP client, so the framework cannot set
    ``max_retries``. ``request_retries=None`` reuses it as-is; a non-None value is
    a configuration error (the framework was asked to configure a client it does
    not own) and is rejected with :class:`ModelRetryConfigurationError`."""
    if bundle.config.protocol == "openai":
        retries = request_retries if isinstance(request_retries, int) else 0
        if retries > 0:
            return ModelBundle.from_config(
                bundle.config, request_retries=retries
            ).model
        return bundle.model
    if request_retries is not None:
        raise ModelRetryConfigurationError(
            f"prebuilt model {bundle.config.model_type!r} cannot be configured with "
            f"request_retries={request_retries!r}; pass request_retries=None so the "
            f"prebuilt model manages its own retry behavior"
        )
    return bundle.model


def _resolved_revision(bundles: "list[ModelBundle]", request_retries: int) -> str:
    """Revision of the resolved model: a stable hash of the ordered candidates'
    own non-secret revisions plus ``request_retries``. Candidate order is
    significant, so reordering the fallback chain changes the revision; each
    candidate's own revision already excludes secrets, so key rotation does
    not."""
    identity = [
        {"revision": b.revision, "request_retries": request_retries}
        for b in bundles
    ]
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


__all__: "list[str]" = [
    "ResolvedModel",
    "ModelResolver",
    "ModelRoutingError",
    "ModelRetryConfigurationError",
]
